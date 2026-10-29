"""D-only learned local writer for the Spark persistent-learner design.

This module is a deterministic CPU engineering fixture for the *expert family*
that freezes the offline feature transforms ``U`` and ``G`` and updates only
the online readout matrix ``D``.  It is not a native Spark run, not a trained
language model, and not evidence of native multimodal or lifelong capability.
The evidence class is recorded in every report.

Math
----
For a fixed feature map

    h(z) = SiLU(U z) * (G z)

and a frozen readout ``R``, the online state produces logits

    logits(x) = R (D h(z(x))) .

The writer machinery ``phi`` proposes a local write target from a causal
training example ``(x_t, y_t)``:

    z_t = tanh(W_ctx [x_t; onehot(y_t)] + b_ctx)
    gain = softplus(gain_log)

Two separately declared functional one-step arms are provided.

Functional SGD (the default arm)::

    p_t   = softmax(R D h(z_t))
    q_t   = onehot(y_t)
    g_D   = R^T (p_t - q_t) h(z_t)^T
    D'    = D - inner_lr * gain * g_D

Functional delta (a lower-cost research arm without an inner softmax CE)::

    v     = W_val z_t + b_val
    h_t   = h(x_t)                       # fixed key from the raw example
    D'    = D + inner_lr * gain * v h_t^T

The outer writer objective uses a *distinct future query* (and, when present,
replay/old-skill queries)::

    J(phi) = CE(R D' h(x_f), y_f)
             + replay_weight * mean_replay CE(R D' h(x_r), y_r)
             + cost_weight * ||D' - D||_F / cost_reference

The cost term is a differentiable write-payload surrogate built from the
measured update delta.  Declared compute cost and wall-clock time are reported
separately; they are not differentiated.

Derivative semantics
--------------------
``functional_sgd`` with ``derivative_mode="exact"`` differentiates through the
inner gradient analytically, including the softmax Hessian term

    H_p = diag(p_t) - p_t p_t^T

which appears because ``z_t`` (and therefore ``p_t``) depends on ``phi``.  The
closed-form vector-Jacobian product is implemented in
:func:`sgd_meta_gradient`.  It matches central finite differences of the full
one-step objective.

``derivative_mode="first_order_truncated"`` stops the gradient through ``p_t``
in the inner write gradient.  It is a declared approximation, not the exact
meta-gradient, and is retained only as a comparison arm.

``functional_delta`` has an exact first-order derivative through ``v``, ``z_t``
and ``gain``; it has no inner softmax term, so it has no truncated variant.

The imperative AdamW reference is explicitly non-differentiable.  Production
uses :class:`replica_lab.optimizers.NativeFusedAdamW`; this module implements
the same imperative update equation in NumPy only as a cost/reference arm and
never backpropagates through it.

One clock
---------
The deployed learner has one online clock: exactly one functional ``D`` update
per episode.  Offline ``phi`` training is represented separately and does not
create a second online timescale.  There is no RSI or multi-clock promotion API.

Sealed evaluation
-----------------
The world suite exposes a sealed split only as closed world identifiers.
:meth:`WorldSuite.episodes` raises :class:`SealedEvaluationClosed` for that
split, and no training or evaluation path opens it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Iterable, Mapping, Sequence

import numpy as np

FORMAT = "replica.learned_local_writer.v1"
REPORT_FORMAT = "replica.learned_local_writer.report.v1"
EVIDENCE_CLASS = "cpu_engineering_fixture_not_native_capability"
SEALED_SPLIT = "sealed"

ARM_SGD = "functional_sgd"
ARM_DELTA = "functional_delta"
ARM_ADAMW_REFERENCE = "imperative_adamw_reference"
FUNCTIONAL_ARMS = (ARM_SGD, ARM_DELTA)
NON_DIFFERENTIABLE_ARMS = (ARM_ADAMW_REFERENCE,)

DERIVATIVE_EXACT = "exact"
DERIVATIVE_TRUNCATED = "first_order_truncated"
DERIVATIVE_MODES = (DERIVATIVE_EXACT, DERIVATIVE_TRUNCATED)

TRAIN_FAMILIES = ("direct", "shift")
DEVELOPMENT_FAMILIES = ("mirror",)
PRIOR_FAMILY = "prior_holdout"

_ARRAY_NAMES = ("W_ctx", "b_ctx", "W_val", "b_val")

_LIMITS = (
    "CPU engineering fixture only; not native Spark training and not a "
    "native-capability claim.",
    "Synthetic worlds only; no teacher-competence, verifier-reliability, "
    "dataset or identity claim is established.",
    "One functional D update per episode; no RSI, no multi-clock promotion.",
    "Sealed evaluation is closed and is never an outer-training query set.",
    "Exact is exact only for this declared fixture; the truncated SGD arm is "
    "a first-order/stop-gradient approximation.",
    "The imperative AdamW reference is non-differentiable and is not used to "
    "train phi.",
    "Measured wall-clock is a local CPU observation, not a resource guarantee.",
    "Parameter changes alone are not reported as useful learning; transfer, "
    "retention, controls and cost are reported separately.",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        scalar = float(arr)
        if scalar >= 0.0:
            return 1.0 / (1.0 + math.exp(-scalar))
        exp = math.exp(scalar)
        return exp / (1.0 + exp)
    out = np.empty_like(arr)
    positive = arr >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp = np.exp(arr[~positive])
    out[~positive] = exp / (1.0 + exp)
    return out


def _softplus(value: float) -> float:
    return float(np.logaddexp(0.0, float(value)))


def _freeze(value: np.ndarray) -> np.ndarray:
    arr = np.array(value, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


def _as_matrix(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    _require(arr.shape == shape, f"{name} must have shape {shape}, got {arr.shape}")
    _require(bool(np.all(np.isfinite(arr))), f"{name} must be finite")
    return arr


def _sha_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _mean_or_none(values: Sequence[float] | None) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


@dataclass(frozen=True)
class WriterShapes:
    """Bounded fixture shapes; all dimensions are positive integers."""

    in_dim: int = 6
    hidden: int = 10
    latent: int = 6
    classes: int = 3

    def __post_init__(self) -> None:
        for name in ("in_dim", "hidden", "latent", "classes"):
            value = getattr(self, name)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value > 0,
                f"{name} must be a positive integer",
            )

    @property
    def ctx_in(self) -> int:
        return self.in_dim + self.classes

    @property
    def context_shape(self) -> tuple[int, int]:
        return (self.in_dim, self.ctx_in)


DEFAULT_SHAPES = WriterShapes()


@dataclass(frozen=True, eq=False)
class FixedBackbone:
    """Offline-fixed U/G/readout plus the initial online D."""

    U: np.ndarray
    G: np.ndarray
    readout: np.ndarray
    D0: np.ndarray
    shapes: WriterShapes

    def __post_init__(self) -> None:
        _require(isinstance(self.shapes, WriterShapes), "shapes must be WriterShapes")
        expected = {
            "U": (self.shapes.hidden, self.shapes.in_dim),
            "G": (self.shapes.hidden, self.shapes.in_dim),
            "readout": (self.shapes.classes, self.shapes.latent),
            "D0": (self.shapes.latent, self.shapes.hidden),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            _require(isinstance(value, np.ndarray), f"{name} must be a numpy array")
            arr = np.array(value, dtype=np.float64, copy=True)
            _require(arr.shape == shape, f"{name} must have shape {shape}")
            _require(bool(np.all(np.isfinite(arr))), f"{name} must be finite")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    @classmethod
    def from_seed(
        cls,
        shapes: WriterShapes = DEFAULT_SHAPES,
        seed: int = 0,
        *,
        D0_scale: float = 0.05,
    ) -> "FixedBackbone":
        """Create deterministic offline features and a small initial D."""
        _require(D0_scale >= 0.0, "D0_scale must be nonnegative")
        rng = np.random.default_rng(seed)
        U = rng.normal(0.0, 0.6, size=(shapes.hidden, shapes.in_dim))
        G = rng.normal(0.0, 0.6, size=(shapes.hidden, shapes.in_dim))
        readout = rng.normal(0.0, 1.0, size=(shapes.classes, shapes.latent))
        readout = readout / math.sqrt(shapes.latent)
        D0 = rng.normal(0.0, D0_scale, size=(shapes.latent, shapes.hidden))
        return cls(U=U, G=G, readout=readout, D0=D0, shapes=shapes)

    def feature(self, z) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        return _feature(self, z)

    def digest(self) -> str:
        return hashlib.sha256(
            "|".join(
                _sha_array(getattr(self, name))
                for name in ("U", "G", "readout", "D0")
            ).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True, eq=False)
class WriterParams:
    """Offline writer machinery phi: context encoder, delta value, gain."""

    W_ctx: np.ndarray
    b_ctx: np.ndarray
    W_val: np.ndarray
    b_val: np.ndarray
    gain_log: float
    shapes: WriterShapes

    def __post_init__(self) -> None:
        _require(isinstance(self.shapes, WriterShapes), "shapes must be WriterShapes")
        expected = {
            "W_ctx": self.shapes.context_shape,
            "b_ctx": (self.shapes.in_dim,),
            "W_val": (self.shapes.latent, self.shapes.in_dim),
            "b_val": (self.shapes.latent,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            _require(isinstance(value, np.ndarray), f"{name} must be a numpy array")
            arr = np.array(value, dtype=np.float64, copy=True)
            _require(arr.shape == shape, f"{name} must have shape {shape}")
            _require(bool(np.all(np.isfinite(arr))), f"{name} must be finite")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        gain_log = float(self.gain_log)
        _require(math.isfinite(gain_log), "gain_log must be finite")
        object.__setattr__(self, "gain_log", gain_log)

    @classmethod
    def initial(
        cls,
        shapes: WriterShapes = DEFAULT_SHAPES,
        seed: int = 0,
    ) -> "WriterParams":
        rng = np.random.default_rng(seed)
        return cls(
            W_ctx=rng.normal(0.0, 0.3, size=shapes.context_shape),
            b_ctx=np.zeros(shapes.in_dim),
            W_val=rng.normal(0.0, 0.3, size=(shapes.latent, shapes.in_dim)),
            b_val=np.zeros(shapes.latent),
            gain_log=0.0,
            shapes=shapes,
        )

    @property
    def gain(self) -> float:
        return _softplus(self.gain_log)

    @property
    def parameter_count(self) -> int:
        return sum(getattr(self, name).size for name in _ARRAY_NAMES) + 1

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.array(getattr(self, name), copy=True) for name in _ARRAY_NAMES}

    def with_arrays(self, **changes) -> "WriterParams":
        values = self.arrays()
        gain_log = changes.pop("gain_log", self.gain_log)
        for name in _ARRAY_NAMES:
            if name in changes:
                values[name] = changes.pop(name)
        if changes:
            raise TypeError(f"unknown writer parameter(s): {sorted(changes)}")
        return WriterParams(
            W_ctx=values["W_ctx"],
            b_ctx=values["b_ctx"],
            W_val=values["W_val"],
            b_val=values["b_val"],
            gain_log=gain_log,
            shapes=self.shapes,
        )

    def norm(self) -> float:
        return _global_norm(self)

    def digest(self) -> str:
        shape_text = "|".join(
            str(value)
            for value in (
                self.shapes.in_dim,
                self.shapes.hidden,
                self.shapes.latent,
                self.shapes.classes,
            )
        )
        return hashlib.sha256(
            shape_text.encode("ascii")
            + b"|"
            + "|".join(_sha_array(getattr(self, name)) for name in _ARRAY_NAMES).encode("ascii")
            + b"|"
            + f"{self.gain_log:.17g}".encode("ascii")
        ).hexdigest()


@dataclass(frozen=True)
class Example:
    """One causal query.  Sealed examples are never materialized by the suite."""

    query_id: str
    world_id: str
    family: str
    split: str
    x: tuple[float, ...]
    label: int

    def __post_init__(self) -> None:
        for name in ("query_id", "world_id", "family", "split"):
            value = getattr(self, name)
            _require(isinstance(value, str) and bool(value), f"{name} must be nonempty")
        _require(isinstance(self.x, tuple) and bool(self.x), "x must be a nonempty tuple")
        _require(
            all(isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in self.x),
            "x must contain finite numbers",
        )
        _require(isinstance(self.label, int) and not isinstance(self.label, bool),
                 "label must be an int")
        _require(self.label >= 0, "label must be nonnegative")

    def vector(self) -> np.ndarray:
        return np.asarray(self.x, dtype=np.float64)


@dataclass(frozen=True)
class Episode:
    """Chronological episode: one write query, future queries, replay queries."""

    episode_id: str
    world_id: str
    family: str
    split: str
    write: Example
    future: tuple[Example, ...]
    replay: tuple[Example, ...]

    def __post_init__(self) -> None:
        _require(isinstance(self.write, Example), "write must be an Example")
        _require(self.write.split == self.split, "write split must match episode split")
        _require(self.write.world_id == self.world_id,
                 "write world must match episode world")
        _require(isinstance(self.future, tuple) and bool(self.future),
                 "future must be a nonempty tuple")
        _require(isinstance(self.replay, tuple), "replay must be a tuple")
        identifiers = [self.write.query_id]
        for example in self.future:
            _require(isinstance(example, Example), "future entries must be Examples")
            _require(example.split == self.split, "future split must match episode")
            _require(example.world_id == self.world_id,
                     "future world must match episode")
            identifiers.append(example.query_id)
        _require(len(set(identifiers)) == len(identifiers),
                 "write and future query ids must be distinct")
        for example in self.replay:
            _require(isinstance(example, Example), "replay entries must be Examples")
            _require(example.query_id not in identifiers,
                     "replay must not duplicate write/future queries")


class SealedEvaluationClosed(RuntimeError):
    """Raised when a caller asks for sealed data."""


@dataclass(frozen=True)
class WorldSuite:
    """Independent synthetic worlds with a permanently closed sealed split."""

    train: tuple[Episode, ...]
    development: tuple[Episode, ...]
    sealed_world_ids: tuple[str, ...]
    prior_world_id: str
    seed: int
    shapes: WriterShapes

    def __post_init__(self) -> None:
        _require(isinstance(self.train, tuple) and bool(self.train),
                 "train must be a nonempty tuple")
        _require(isinstance(self.development, tuple),
                 "development must be a tuple")
        online_worlds = {episode.world_id for episode in self.train + self.development}
        _require(not (online_worlds & set(self.sealed_world_ids)),
                 "sealed world ids must not appear in online episodes")
        _require(self.prior_world_id not in online_worlds,
                 "prior world must not be an adaptation world")
        _require(self.prior_world_id not in set(self.sealed_world_ids),
                 "prior world must not be sealed")

    def episodes(self, split: str) -> tuple[Episode, ...]:
        if split == "train":
            return self.train
        if split == "development":
            return self.development
        if split == SEALED_SPLIT:
            raise SealedEvaluationClosed(
                "sealed evaluation is closed; it is not an outer-training or "
                "development query set"
            )
        raise ValueError(f"unknown split: {split}")

    @property
    def family_splits(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": tuple(sorted({episode.family for episode in self.train})),
            "development": tuple(sorted({episode.family for episode in self.development})),
            "sealed": (),
        }

    def metadata(self) -> dict:
        return {
            "seed": self.seed,
            "shapes": {
                "in_dim": self.shapes.in_dim,
                "hidden": self.shapes.hidden,
                "latent": self.shapes.latent,
                "classes": self.shapes.classes,
            },
            "train_episodes": len(self.train),
            "development_episodes": len(self.development),
            "train_worlds": tuple(sorted({episode.world_id for episode in self.train})),
            "development_worlds": tuple(
                sorted({episode.world_id for episode in self.development})
            ),
            "sealed_world_count": len(self.sealed_world_ids),
            "sealed_access": "closed",
            "families": self.family_splits,
        }


@dataclass(frozen=True)
class UpdateCost:
    """Declared compute and measured local update cost; no native claim."""

    arm: str
    inner_steps: int
    scalar_ops: int
    meta_scalar_ops: int
    changed_entries: int
    delta_norm: float
    measured_seconds: float

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "inner_steps": self.inner_steps,
            "declared_scalar_ops": self.scalar_ops,
            "declared_meta_scalar_ops": self.meta_scalar_ops,
            "changed_entries": self.changed_entries,
            "delta_norm": self.delta_norm,
            "measured_seconds": self.measured_seconds,
            "measurement_scope": "cpu_fixture_not_native",
        }


@dataclass(frozen=True, eq=False)
class WriteStep:
    """The forward one-step write and the quantities needed for its VJP."""

    arm: str
    inner_lr: float
    D: np.ndarray
    D_prime: np.ndarray
    delta: np.ndarray
    gain: float
    z: np.ndarray
    h: np.ndarray
    p: np.ndarray | None
    q: np.ndarray | None
    value: np.ndarray | None
    inner_grad: np.ndarray | None
    context_input: np.ndarray
    dtanh: np.ndarray
    feature_cache: tuple[np.ndarray, ...]
    cost: UpdateCost


@dataclass(frozen=True, eq=False)
class OuterResult:
    """One outer objective evaluation and its writer-machinery gradient."""

    arm: str
    derivative_mode: str
    value: float
    future_loss: float
    replay_loss: float
    cost_term: float
    cost_norm: float
    future_accuracy: float
    replay_accuracy: float
    upstream_d: np.ndarray
    grad: WriterParams
    step: WriteStep

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "derivative_mode": self.derivative_mode,
            "value": self.value,
            "future_loss": self.future_loss,
            "replay_loss": self.replay_loss,
            "cost_term": self.cost_term,
            "cost_norm": self.cost_norm,
            "future_accuracy": self.future_accuracy,
            "replay_accuracy": self.replay_accuracy,
            "grad_norm": self.grad.norm(),
        }


# ---------------------------------------------------------------------------
# Core differentiable fixture
# ---------------------------------------------------------------------------

def _feature(backbone: FixedBackbone, z) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    z = _as_matrix(z, (backbone.shapes.in_dim,), "z")
    a = backbone.U @ z
    sig = _sigmoid(a)
    dsilu = sig * (1.0 + a * (1.0 - sig))
    silu = a * sig
    b = backbone.G @ z
    h = silu * b
    return h, (a, sig, dsilu, silu, b)


def _feature_vjp(
    backbone: FixedBackbone,
    cache: tuple[np.ndarray, ...],
    upstream_h: np.ndarray,
) -> np.ndarray:
    _a, _sig, dsilu, silu, b = cache
    upstream_h = _as_matrix(upstream_h, (backbone.shapes.hidden,), "upstream_h")
    return (upstream_h * b * dsilu) @ backbone.U + (upstream_h * silu) @ backbone.G


def softmax(logits) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    _require(logits.ndim == 1 and logits.size > 0, "logits must be a nonempty vector")
    _require(bool(np.all(np.isfinite(logits))), "logits must be finite")
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def one_hot(label: int, classes: int) -> np.ndarray:
    _require(isinstance(classes, int) and classes > 0, "classes must be positive")
    _require(isinstance(label, int) and 0 <= label < classes,
             f"label must be in [0, {classes})")
    value = np.zeros(classes, dtype=np.float64)
    value[label] = 1.0
    return value


def cross_entropy(probabilities, label: int) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    _require(probabilities.ndim == 1 and probabilities.size > 0,
             "probabilities must be a nonempty vector")
    _require(0 <= label < probabilities.size, "label out of range")
    return float(-math.log(max(float(probabilities[label]), 1e-300)))


def _write_context(
    phi: WriterParams,
    example: Example,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shapes = phi.shapes
    x = _as_matrix(example.vector(), (shapes.in_dim,), "example.x")
    _require(example.label < shapes.classes, "example label outside class range")
    context_input = np.concatenate([x, one_hot(example.label, shapes.classes)])
    z0 = phi.W_ctx @ context_input + phi.b_ctx
    z = np.tanh(z0)
    dtanh = 1.0 - z * z
    return z, z0, dtanh, context_input


def _dispatch_step(
    backbone: FixedBackbone,
    phi: WriterParams,
    D,
    write: Example,
    *,
    arm: str,
    inner_lr: float,
    measure_cost: bool = True,
) -> WriteStep:
    if arm == ARM_SGD:
        return functional_sgd_step(
            backbone, phi, D, write, inner_lr=inner_lr, measure_cost=measure_cost
        )
    if arm == ARM_DELTA:
        return functional_delta_step(
            backbone, phi, D, write, inner_lr=inner_lr, measure_cost=measure_cost
        )
    raise ValueError(f"unsupported functional arm: {arm}")


def _update_cost(
    arm: str,
    delta: np.ndarray,
    elapsed: float,
    shapes: WriterShapes,
) -> UpdateCost:
    return UpdateCost(
        arm=arm,
        inner_steps=1,
        scalar_ops=declared_update_ops(arm, shapes),
        meta_scalar_ops=0,
        changed_entries=int(np.count_nonzero(np.abs(delta) > 1e-12)),
        delta_norm=float(np.linalg.norm(delta)),
        measured_seconds=float(max(elapsed, 0.0)),
    )


def functional_sgd_step(
    backbone: FixedBackbone,
    phi: WriterParams,
    D,
    write: Example,
    *,
    inner_lr: float = 0.7,
    measure_cost: bool = True,
) -> WriteStep:
    """One exact functional SGD write: ``D' = D - lr * gain * R^T(p-q)h^T``."""
    _require(isinstance(phi, WriterParams), "phi must be WriterParams")
    _require(phi.shapes == backbone.shapes, "phi/backbone shapes must match")
    _require(isinstance(write, Example), "write must be an Example")
    _require(math.isfinite(inner_lr) and inner_lr > 0.0, "inner_lr must be positive")
    D = _as_matrix(D, (backbone.shapes.latent, backbone.shapes.hidden), "D")

    started = time.perf_counter() if measure_cost else 0.0
    z, _z0, dtanh, context_input = _write_context(phi, write)
    h, cache = _feature(backbone, z)
    probabilities = softmax(backbone.readout @ (D @ h))
    target = one_hot(write.label, backbone.shapes.classes)
    inner_grad = np.outer(backbone.readout.T @ (probabilities - target), h)
    gain = phi.gain
    delta = -inner_lr * gain * inner_grad
    D_prime = D + delta
    elapsed = time.perf_counter() - started if measure_cost else 0.0

    return WriteStep(
        arm=ARM_SGD,
        inner_lr=float(inner_lr),
        D=_freeze(D),
        D_prime=_freeze(D_prime),
        delta=_freeze(delta),
        gain=gain,
        z=_freeze(z),
        h=_freeze(h),
        p=_freeze(probabilities),
        q=_freeze(target),
        value=None,
        inner_grad=_freeze(inner_grad),
        context_input=_freeze(context_input),
        dtanh=_freeze(dtanh),
        feature_cache=tuple(_freeze(value) for value in cache),
        cost=_update_cost(ARM_SGD, delta, elapsed, backbone.shapes),
    )


def functional_delta_step(
    backbone: FixedBackbone,
    phi: WriterParams,
    D,
    write: Example,
    *,
    inner_lr: float = 0.7,
    measure_cost: bool = True,
) -> WriteStep:
    """One functional delta write: ``D' = D + lr * gain * v h_t^T``."""
    _require(isinstance(phi, WriterParams), "phi must be WriterParams")
    _require(phi.shapes == backbone.shapes, "phi/backbone shapes must match")
    _require(isinstance(write, Example), "write must be an Example")
    _require(math.isfinite(inner_lr) and inner_lr > 0.0, "inner_lr must be positive")
    D = _as_matrix(D, (backbone.shapes.latent, backbone.shapes.hidden), "D")

    started = time.perf_counter() if measure_cost else 0.0
    z, _z0, dtanh, context_input = _write_context(phi, write)
    value = phi.W_val @ z + phi.b_val
    h, cache = _feature(backbone, write.vector())
    gain = phi.gain
    delta = inner_lr * gain * np.outer(value, h)
    D_prime = D + delta
    elapsed = time.perf_counter() - started if measure_cost else 0.0

    return WriteStep(
        arm=ARM_DELTA,
        inner_lr=float(inner_lr),
        D=_freeze(D),
        D_prime=_freeze(D_prime),
        delta=_freeze(delta),
        gain=gain,
        z=_freeze(z),
        h=_freeze(h),
        p=None,
        q=None,
        value=_freeze(value),
        inner_grad=None,
        context_input=_freeze(context_input),
        dtanh=_freeze(dtanh),
        feature_cache=tuple(_freeze(value) for value in cache),
        cost=_update_cost(ARM_DELTA, delta, elapsed, backbone.shapes),
    )


# ---------------------------------------------------------------------------
# Analytic meta-gradients (exact versus declared truncation)
# ---------------------------------------------------------------------------

def sgd_meta_gradient(
    backbone: FixedBackbone,
    phi: WriterParams,
    step: WriteStep,
    upstream_D,
    *,
    derivative_mode: str = DERIVATIVE_EXACT,
) -> WriterParams:
    """VJP of ``<upstream_D, D'>`` for the SGD arm.

    Exact mode includes the softmax Hessian term
    ``u_p = D^T R^T H_p R (upstream_D h)``.  Truncated mode drops ``u_p`` by
    treating the write-query probabilities as a constant, which is a declared
    first-order/stop-gradient approximation.
    """
    _require(step.arm == ARM_SGD, "step is not the functional_sgd arm")
    _require(derivative_mode in DERIVATIVE_MODES,
             f"derivative_mode must be one of {DERIVATIVE_MODES}")
    _require(step.p is not None and step.q is not None and step.inner_grad is not None,
             "SGD step is missing probabilities")
    upstream_D = _as_matrix(
        upstream_D,
        (backbone.shapes.latent, backbone.shapes.hidden),
        "upstream_D",
    )

    probabilities = step.p
    target = step.q
    h = step.h
    n = backbone.shapes

    base = backbone.readout.T @ (probabilities - target)
    u_direct = upstream_D.T @ base
    c = backbone.readout @ (upstream_D @ h)
    hessian_c = probabilities * c - probabilities * float(probabilities @ c)
    u_hessian = step.D.T @ (backbone.readout.T @ hessian_c)
    upstream_h = u_direct if derivative_mode == DERIVATIVE_TRUNCATED else u_direct + u_hessian
    upstream_z = _feature_vjp(backbone, step.feature_cache, upstream_h)
    upstream_z0 = upstream_z * step.dtanh

    scale = -step.inner_lr * step.gain
    grad_W_ctx = scale * np.outer(upstream_z0, step.context_input)
    grad_b_ctx = scale * upstream_z0
    grad_gain_log = (
        -step.inner_lr
        * float(_sigmoid(phi.gain_log))
        * float(np.sum(upstream_D * step.inner_grad))
    )

    return phi.with_arrays(
        W_ctx=grad_W_ctx,
        b_ctx=grad_b_ctx,
        W_val=np.zeros((n.latent, n.in_dim)),
        b_val=np.zeros(n.latent),
        gain_log=grad_gain_log,
    )


def delta_meta_gradient(
    backbone: FixedBackbone,
    phi: WriterParams,
    step: WriteStep,
    upstream_D,
    *,
    derivative_mode: str = DERIVATIVE_EXACT,
) -> WriterParams:
    """Exact first-order VJP for the functional delta arm.

    The delta arm has no inner softmax CE, so there is no truncated variant.
    """
    _require(step.arm == ARM_DELTA, "step is not the functional_delta arm")
    _require(derivative_mode == DERIVATIVE_EXACT,
             "functional_delta has only an exact first-order derivative")
    _require(step.value is not None, "delta step is missing its value")
    upstream_D = _as_matrix(
        upstream_D,
        (backbone.shapes.latent, backbone.shapes.hidden),
        "upstream_D",
    )

    upstream_value = upstream_D @ step.h
    scale = step.inner_lr * step.gain
    upstream_z = phi.W_val.T @ upstream_value
    upstream_z0 = upstream_z * step.dtanh
    grad_W_ctx = scale * np.outer(upstream_z0, step.context_input)
    grad_b_ctx = scale * upstream_z0
    grad_W_val = scale * np.outer(upstream_value, step.z)
    grad_b_val = scale * upstream_value
    gain_dot = float(np.sum(upstream_D * step.delta))
    grad_gain_log = float(_sigmoid(phi.gain_log)) * gain_dot / max(step.gain, 1e-12)

    return phi.with_arrays(
        W_ctx=grad_W_ctx,
        b_ctx=grad_b_ctx,
        W_val=grad_W_val,
        b_val=grad_b_val,
        gain_log=grad_gain_log,
    )


# ---------------------------------------------------------------------------
# Outer objective, transfer, retention and cost
# ---------------------------------------------------------------------------

def declared_update_ops(arm: str, shapes: WriterShapes = DEFAULT_SHAPES) -> int:
    """Declared scalar multiply-add model for one functional update."""
    in_dim, hidden, latent, classes = (
        shapes.in_dim, shapes.hidden, shapes.latent, shapes.classes
    )
    context = in_dim * shapes.ctx_in + shapes.ctx_in
    feature = 2 * hidden * in_dim + 6 * hidden
    if arm in (ARM_SGD, ARM_ADAMW_REFERENCE):
        forward = context + feature + latent * hidden + classes * latent + 3 * classes
        gradient = classes * latent + latent * hidden
        update = latent * hidden
        value = forward + gradient + update
        if arm == ARM_ADAMW_REFERENCE:
            value += 6 * latent * hidden
        return int(value)
    if arm == ARM_DELTA:
        return int(context + feature + latent * in_dim + latent * hidden)
    raise ValueError(f"unknown arm: {arm}")


def declared_meta_ops(
    arm: str,
    derivative_mode: str,
    shapes: WriterShapes = DEFAULT_SHAPES,
) -> int:
    """Declared scalar op model for the writer VJP (not a measurement)."""
    in_dim, hidden, latent, classes = (
        shapes.in_dim, shapes.hidden, shapes.latent, shapes.classes
    )
    if arm == ARM_SGD:
        if derivative_mode == DERIVATIVE_EXACT:
            return int(
                classes * latent + latent * hidden + classes * classes
                + hidden * in_dim * 2 + latent * classes + hidden
            )
        if derivative_mode == DERIVATIVE_TRUNCATED:
            return int(classes * latent + latent * hidden + hidden * in_dim * 2 + hidden)
        raise ValueError(f"unknown derivative mode: {derivative_mode}")
    if arm == ARM_DELTA:
        return int(latent * in_dim + in_dim * shapes.ctx_in + latent * hidden)
    raise ValueError(f"unknown arm: {arm}")


def _query_metrics(
    backbone: FixedBackbone,
    D: np.ndarray,
    examples: Sequence[Example],
) -> dict | None:
    examples = tuple(examples)
    if not examples:
        return None
    losses: list[float] = []
    correct = 0
    for example in examples:
        _require(example.label < backbone.shapes.classes,
                 "example label outside class range")
        h, _cache = _feature(backbone, example.vector())
        probabilities = softmax(backbone.readout @ (D @ h))
        losses.append(cross_entropy(probabilities, example.label))
        correct += int(np.argmax(probabilities) == example.label)
    return {
        "ce": float(np.mean(losses)),
        "accuracy": correct / len(examples),
        "count": len(examples),
    }


def _upstream_from_examples(
    backbone: FixedBackbone,
    D: np.ndarray,
    examples: Sequence[Example],
    weight: float,
) -> tuple[float, np.ndarray, float | None]:
    examples = tuple(examples)
    if not examples:
        return 0.0, np.zeros_like(D), None
    losses: list[float] = []
    upstream = np.zeros_like(D)
    correct = 0
    for example in examples:
        h, _cache = _feature(backbone, example.vector())
        probabilities = softmax(backbone.readout @ (D @ h))
        losses.append(cross_entropy(probabilities, example.label))
        upstream += np.outer(
            backbone.readout.T
            @ (probabilities - one_hot(example.label, backbone.shapes.classes)),
            h,
        )
        correct += int(np.argmax(probabilities) == example.label)
    count = len(examples)
    return (
        float(np.mean(losses)),
        (weight / count) * upstream,
        correct / count,
    )


def _outer_parts(
    backbone: FixedBackbone,
    phi: WriterParams,
    episode: Episode,
    *,
    arm: str,
    inner_lr: float,
    replay_weight: float,
    cost_weight: float,
    cost_reference: float,
    measure_cost: bool = True,
) -> tuple[float, float, float, float, float, float, float, np.ndarray, WriteStep]:
    _require(isinstance(episode, Episode), "episode must be an Episode")
    _require(episode.split != SEALED_SPLIT, "sealed episodes are closed")
    _require(math.isfinite(replay_weight) and replay_weight >= 0.0,
             "replay_weight must be nonnegative")
    _require(math.isfinite(cost_weight) and cost_weight >= 0.0,
             "cost_weight must be nonnegative")
    _require(math.isfinite(cost_reference) and cost_reference > 0.0,
             "cost_reference must be positive")

    step = _dispatch_step(
        backbone, phi, backbone.D0, episode.write,
        arm=arm, inner_lr=inner_lr, measure_cost=measure_cost,
    )
    future_loss, future_up, future_accuracy = _upstream_from_examples(
        backbone, step.D_prime, episode.future, 1.0
    )
    replay_loss, replay_up, replay_accuracy = _upstream_from_examples(
        backbone, step.D_prime, episode.replay, replay_weight
    )
    cost_norm = step.cost.delta_norm
    cost_term = cost_weight * cost_norm / cost_reference
    upstream_D = future_up + replay_up
    if cost_weight > 0.0 and cost_norm > 0.0:
        upstream_D = upstream_D + (cost_weight / cost_reference) * (step.delta / cost_norm)
    value = future_loss + replay_weight * replay_loss + cost_term
    return (
        float(value),
        future_loss,
        replay_loss,
        float(cost_term),
        float(cost_norm),
        float(future_accuracy if future_accuracy is not None else 0.0),
        float(replay_accuracy if replay_accuracy is not None else 0.0),
        upstream_D,
        step,
    )


def outer_value_and_gradient(
    backbone: FixedBackbone,
    phi: WriterParams,
    episode: Episode,
    *,
    arm: str = ARM_SGD,
    inner_lr: float = 0.7,
    replay_weight: float = 0.5,
    cost_weight: float = 0.1,
    cost_reference: float = 0.5,
    derivative_mode: str = DERIVATIVE_EXACT,
) -> OuterResult:
    """Evaluate the outer objective and its analytic writer gradient."""
    _require(arm in FUNCTIONAL_ARMS, f"arm must be one of {FUNCTIONAL_ARMS}")
    _require(derivative_mode in DERIVATIVE_MODES,
             f"derivative_mode must be one of {DERIVATIVE_MODES}")
    if arm == ARM_DELTA:
        _require(derivative_mode == DERIVATIVE_EXACT,
                 "functional_delta has only an exact first-order derivative")
    parts = _outer_parts(
        backbone, phi, episode,
        arm=arm,
        inner_lr=inner_lr,
        replay_weight=replay_weight,
        cost_weight=cost_weight,
        cost_reference=cost_reference,
        measure_cost=True,
    )
    (
        value, future_loss, replay_loss, cost_term, cost_norm,
        future_accuracy, replay_accuracy, upstream_D, step,
    ) = parts
    if arm == ARM_SGD:
        grad = sgd_meta_gradient(
            backbone, phi, step, upstream_D, derivative_mode=derivative_mode
        )
    else:
        grad = delta_meta_gradient(backbone, phi, step, upstream_D)
    return OuterResult(
        arm=arm,
        derivative_mode=derivative_mode,
        value=value,
        future_loss=future_loss,
        replay_loss=replay_loss,
        cost_term=cost_term,
        cost_norm=cost_norm,
        future_accuracy=future_accuracy,
        replay_accuracy=replay_accuracy,
        upstream_d=_freeze(upstream_D),
        grad=grad,
        step=step,
    )


def _outer_value(
    backbone: FixedBackbone,
    phi: WriterParams,
    episode: Episode,
    *,
    arm: str,
    inner_lr: float,
    replay_weight: float,
    cost_weight: float,
    cost_reference: float,
) -> float:
    return _outer_parts(
        backbone, phi, episode,
        arm=arm,
        inner_lr=inner_lr,
        replay_weight=replay_weight,
        cost_weight=cost_weight,
        cost_reference=cost_reference,
        measure_cost=False,
    )[0]


def finite_difference_gradient_check(
    backbone: FixedBackbone,
    phi: WriterParams,
    episode: Episode,
    *,
    arm: str = ARM_SGD,
    inner_lr: float = 0.7,
    replay_weight: float = 0.5,
    cost_weight: float = 0.1,
    cost_reference: float = 0.5,
    derivative_mode: str = DERIVATIVE_EXACT,
    epsilon: float = 1e-6,
) -> dict:
    """Central finite-difference check of the analytic writer VJP."""
    _require(math.isfinite(epsilon) and epsilon > 0.0, "epsilon must be positive")
    analytic = outer_value_and_gradient(
        backbone, phi, episode,
        arm=arm,
        inner_lr=inner_lr,
        replay_weight=replay_weight,
        cost_weight=cost_weight,
        cost_reference=cost_reference,
        derivative_mode=derivative_mode,
    ).grad
    fields: dict[str, dict[str, float]] = {}
    worst_error = 0.0
    worst_gradient = 0.0
    for name in _ARRAY_NAMES:
        base = getattr(phi, name)
        fd = np.zeros_like(base)
        for index in np.ndindex(base.shape):
            plus = np.array(base, copy=True)
            minus = np.array(base, copy=True)
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_phi = phi.with_arrays(**{name: plus})
            minus_phi = phi.with_arrays(**{name: minus})
            fd[index] = (
                _outer_value(
                    backbone, plus_phi, episode,
                    arm=arm, inner_lr=inner_lr, replay_weight=replay_weight,
                    cost_weight=cost_weight, cost_reference=cost_reference,
                )
                - _outer_value(
                    backbone, minus_phi, episode,
                    arm=arm, inner_lr=inner_lr, replay_weight=replay_weight,
                    cost_weight=cost_weight, cost_reference=cost_reference,
                )
            ) / (2.0 * epsilon)
        error = float(np.max(np.abs(fd - getattr(analytic, name))))
        gradient = float(np.max(np.abs(getattr(analytic, name))))
        fields[name] = {
            "max_abs_error": error,
            "max_abs_gradient": gradient,
        }
        worst_error = max(worst_error, error)
        worst_gradient = max(worst_gradient, gradient)

    plus_phi = phi.with_arrays(gain_log=phi.gain_log + epsilon)
    minus_phi = phi.with_arrays(gain_log=phi.gain_log - epsilon)
    fd_gain = (
        _outer_value(
            backbone, plus_phi, episode,
            arm=arm, inner_lr=inner_lr, replay_weight=replay_weight,
            cost_weight=cost_weight, cost_reference=cost_reference,
        )
        - _outer_value(
            backbone, minus_phi, episode,
            arm=arm, inner_lr=inner_lr, replay_weight=replay_weight,
            cost_weight=cost_weight, cost_reference=cost_reference,
        )
    ) / (2.0 * epsilon)
    gain_error = abs(fd_gain - analytic.gain_log)
    fields["gain_log"] = {
        "max_abs_error": float(gain_error),
        "max_abs_gradient": float(abs(analytic.gain_log)),
    }
    worst_error = max(worst_error, float(gain_error))
    worst_gradient = max(worst_gradient, float(abs(analytic.gain_log)))

    return {
        "format": FORMAT,
        "arm": arm,
        "derivative_mode": derivative_mode,
        "epsilon": float(epsilon),
        "max_abs_error": float(worst_error),
        "max_abs_gradient": float(worst_gradient),
        "fields": fields,
        "second_order_term_included": bool(
            arm == ARM_SGD and derivative_mode == DERIVATIVE_EXACT
        ),
        "declared_approximation": bool(
            arm == ARM_SGD and derivative_mode == DERIVATIVE_TRUNCATED
        ),
        "scope": "cpu_engineering_fixture_not_native_capability",
    }


# ---------------------------------------------------------------------------
# Imperative AdamW reference (deliberately non-differentiable)
# ---------------------------------------------------------------------------

def adamw_reference_step(
    backbone: FixedBackbone,
    phi: WriterParams,
    D,
    write: Example,
    *,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    state: Mapping | None = None,
    step_index: int = 1,
) -> tuple[np.ndarray, UpdateCost, dict]:
    """Equation-level imperative AdamW reference; not differentiable.

    Production uses NativeFusedAdamW.  This NumPy reference exists only to
    make the cost/differentiability contrast explicit and is never used to
    train ``phi``.
    """
    _require(0.0 <= beta1 < 1.0, "beta1 must be in [0, 1)")
    _require(0.0 <= beta2 < 1.0, "beta2 must be in [0, 1)")
    _require(eps > 0.0, "eps must be positive")
    _require(lr > 0.0 and math.isfinite(lr), "lr must be positive")
    _require(step_index >= 1, "step_index must be at least one")
    D = _as_matrix(D, (backbone.shapes.latent, backbone.shapes.hidden), "D")
    started = time.perf_counter()

    z, _z0, _dtanh, _context_input = _write_context(phi, write)
    h, _cache = _feature(backbone, z)
    probabilities = softmax(backbone.readout @ (D @ h))
    gradient = np.outer(
        backbone.readout.T
        @ (probabilities - one_hot(write.label, backbone.shapes.classes)),
        h,
    )
    if state is None:
        m = np.zeros_like(D)
        v = np.zeros_like(D)
    else:
        m = np.array(state["m"], dtype=np.float64, copy=True)
        v = np.array(state["v"], dtype=np.float64, copy=True)
    m = beta1 * m + (1.0 - beta1) * gradient
    v = beta2 * v + (1.0 - beta2) * gradient * gradient
    m_hat = m / (1.0 - beta1**step_index)
    v_hat = v / (1.0 - beta2**step_index)
    update = m_hat / (np.sqrt(v_hat) + eps) + weight_decay * D
    D_prime = D - lr * update
    elapsed = time.perf_counter() - started
    delta = D_prime - D
    cost = _update_cost(ARM_ADAMW_REFERENCE, delta, elapsed, backbone.shapes)
    return (
        _freeze(D_prime),
        cost,
        {"m": _freeze(m), "v": _freeze(v), "step_index": step_index + 1},
    )


# ---------------------------------------------------------------------------
# Synthetic worlds: independent train/development families and closed sealed ids
# ---------------------------------------------------------------------------

def _world_examples(
    rng: np.random.Generator,
    shapes: WriterShapes,
    *,
    world_id: str,
    family: str,
    base_label: int,
    split: str,
    count: int,
) -> list[Example]:
    prototype = rng.normal(0.0, 1.0, size=shapes.in_dim)
    prototype = prototype / np.linalg.norm(prototype)
    if family == "direct":
        label = base_label % shapes.classes
    elif family == "shift":
        label = (base_label + 1) % shapes.classes
    elif family == "mirror":
        label = (shapes.classes - 1 - base_label) % shapes.classes
    elif family == PRIOR_FAMILY:
        label = base_label % shapes.classes
    else:
        raise ValueError(f"unknown family: {family}")
    examples: list[Example] = []
    for index in range(count):
        x = prototype * 1.7 + rng.normal(0.0, 0.25, size=shapes.in_dim)
        examples.append(
            Example(
                query_id=f"{world_id}-q{index:02d}",
                world_id=world_id,
                family=family,
                split=split,
                x=tuple(float(value) for value in x),
                label=int(label),
            )
        )
    return examples


def _episode(
    world_id: str,
    family: str,
    split: str,
    examples: Sequence[Example],
    replay: Sequence[Example],
) -> Episode:
    examples = tuple(examples)
    _require(len(examples) >= 2, "an episode needs a write and at least one future query")
    return Episode(
        episode_id=f"{split}-{world_id}",
        world_id=world_id,
        family=family,
        split=split,
        write=examples[0],
        future=tuple(examples[1:]),
        replay=tuple(replay),
    )


def make_world_suite(
    *,
    seed: int = 0,
    shapes: WriterShapes = DEFAULT_SHAPES,
    train_worlds: int = 4,
    development_worlds: int = 3,
    examples_per_world: int = 6,
    replay_examples: int = 4,
) -> WorldSuite:
    """Build independent synthetic worlds with distinct future queries."""
    _require(isinstance(shapes, WriterShapes), "shapes must be WriterShapes")
    for name, value in (
        ("train_worlds", train_worlds),
        ("development_worlds", development_worlds),
        ("examples_per_world", examples_per_world),
        ("replay_examples", replay_examples),
    ):
        _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                 f"{name} must be a positive integer")
    _require(examples_per_world >= 2,
             "examples_per_world must leave a distinct future query")
    rng = np.random.default_rng(seed)

    prior_world_id = f"prior-{seed:04d}"
    prior_examples = _world_examples(
        rng, shapes,
        world_id=prior_world_id,
        family=PRIOR_FAMILY,
        base_label=shapes.classes - 1,
        split="prior",
        count=max(replay_examples, 4),
    )
    replay_pool = tuple(prior_examples[:replay_examples])

    train: list[Episode] = []
    for index in range(train_worlds):
        family = TRAIN_FAMILIES[index % len(TRAIN_FAMILIES)]
        world_id = f"train-{index:02d}-{family}"
        examples = _world_examples(
            rng, shapes,
            world_id=world_id,
            family=family,
            base_label=index % shapes.classes,
            split="train",
            count=examples_per_world,
        )
        train.append(_episode(world_id, family, "train", examples, replay_pool))

    development: list[Episode] = []
    for index in range(development_worlds):
        family = DEVELOPMENT_FAMILIES[index % len(DEVELOPMENT_FAMILIES)]
        world_id = f"development-{index:02d}-{family}"
        examples = _world_examples(
            rng, shapes,
            world_id=world_id,
            family=family,
            base_label=(index + 1) % shapes.classes,
            split="development",
            count=examples_per_world,
        )
        development.append(_episode(world_id, family, "development", examples, replay_pool))

    sealed_world_ids = tuple(f"sealed-{index:02d}" for index in range(3))
    return WorldSuite(
        train=tuple(train),
        development=tuple(development),
        sealed_world_ids=sealed_world_ids,
        prior_world_id=prior_world_id,
        seed=seed,
        shapes=shapes,
    )


# ---------------------------------------------------------------------------
# Evaluation and writer training
# ---------------------------------------------------------------------------

def _aggregate(rows: Sequence[dict], path: Sequence[str]) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row
        for key in path:
            if value is None:
                break
            value = value[key]
        if value is not None:
            values.append(float(value))
    return _mean_or_none(values)


def evaluate_writer(
    backbone: FixedBackbone,
    phi: WriterParams,
    episodes: Sequence[Episode],
    *,
    arm: str = ARM_SGD,
    inner_lr: float = 0.7,
    replay_weight: float = 0.5,
    cost_weight: float = 0.1,
    cost_reference: float = 0.5,
    derivative_mode: str = DERIVATIVE_EXACT,
    min_gain: float = 1e-3,
    max_retention_loss: float = 0.05,
) -> dict:
    """Evaluate transfer, retention, controls, useful learning and cost."""
    _require(arm in FUNCTIONAL_ARMS, f"arm must be one of {FUNCTIONAL_ARMS}")
    episodes = tuple(episodes)
    _require(bool(episodes), "at least one evaluation episode is required")
    for episode in episodes:
        _require(episode.split != SEALED_SPLIT, "sealed episodes are closed")

    rows: list[dict] = []
    for index, episode in enumerate(episodes):
        no_write_future = _query_metrics(backbone, backbone.D0, episode.future)
        step = _dispatch_step(
            backbone, phi, backbone.D0, episode.write,
            arm=arm, inner_lr=inner_lr, measure_cost=True,
        )
        write_future = _query_metrics(backbone, step.D_prime, episode.future)
        no_write_replay = _query_metrics(backbone, backbone.D0, episode.replay)
        write_replay = _query_metrics(backbone, step.D_prime, episode.replay)

        wrong_future = None
        if len(episodes) > 1:
            wrong_write = episodes[(index + 1) % len(episodes)].write
            wrong_step = _dispatch_step(
                backbone, phi, backbone.D0, wrong_write,
                arm=arm, inner_lr=inner_lr, measure_cost=True,
            )
            wrong_future = _query_metrics(backbone, wrong_step.D_prime, episode.future)

        shuffled_write = Example(
            query_id=f"{episode.write.query_id}-shuffled",
            world_id=episode.write.world_id,
            family=episode.write.family,
            split=episode.write.split,
            x=episode.write.x,
            label=(episode.write.label + 1) % backbone.shapes.classes,
        )
        shuffled_step = _dispatch_step(
            backbone, phi, backbone.D0, shuffled_write,
            arm=arm, inner_lr=inner_lr, measure_cost=True,
        )
        shuffled_future = _query_metrics(backbone, shuffled_step.D_prime, episode.future)

        rows.append(
            {
                "episode_id": episode.episode_id,
                "world_id": episode.world_id,
                "family": episode.family,
                "future_no_write": no_write_future,
                "future_write": write_future,
                "replay_no_write": no_write_replay,
                "replay_write": write_replay,
                "wrong_world": wrong_future,
                "shuffled_label": shuffled_future,
                "cost": step.cost.as_dict(),
                "delta_norm": step.cost.delta_norm,
                "gain": step.gain,
            }
        )

    future_before = _aggregate(rows, ("future_no_write", "ce"))
    future_after = _aggregate(rows, ("future_write", "ce"))
    future_accuracy_before = _aggregate(rows, ("future_no_write", "accuracy"))
    future_accuracy_after = _aggregate(rows, ("future_write", "accuracy"))
    wrong_world_ce = _aggregate(rows, ("wrong_world", "ce"))
    shuffled_ce = _aggregate(rows, ("shuffled_label", "ce"))
    replay_before = _aggregate(rows, ("replay_no_write", "ce"))
    replay_after = _aggregate(rows, ("replay_write", "ce"))
    replay_accuracy_before = _aggregate(rows, ("replay_no_write", "accuracy"))
    replay_accuracy_after = _aggregate(rows, ("replay_write", "accuracy"))

    future_improvement = (
        None if future_before is None or future_after is None
        else future_before - future_after
    )
    replay_change = (
        None if replay_before is None or replay_after is None
        else replay_after - replay_before
    )

    criteria: dict[str, bool] = {
        "future_ce_improves": bool(
            future_improvement is not None and future_improvement >= min_gain
        ),
        "beats_wrong_world": bool(
            future_after is not None and wrong_world_ce is not None
            and future_after <= wrong_world_ce - min_gain
        ),
        "beats_shuffled_label": bool(
            future_after is not None and shuffled_ce is not None
            and future_after <= shuffled_ce - min_gain
        ),
        "retention_within_tolerance": bool(
            replay_change is not None and replay_change <= max_retention_loss
        ),
    }
    useful_passed = all(
        criteria[name]
        for name in ("future_ce_improves", "beats_wrong_world", "beats_shuffled_label")
    )

    update_costs = [row["cost"] for row in rows]
    return {
        "transfer": {
            "episodes": len(rows),
            "worlds": tuple(sorted({row["world_id"] for row in rows})),
            "future_ce_before": future_before,
            "future_ce_after": future_after,
            "future_ce_improvement": future_improvement,
            "future_accuracy_before": future_accuracy_before,
            "future_accuracy_after": future_accuracy_after,
            "controls": {
                "no_write": future_before,
                "wrong_world": wrong_world_ce,
                "shuffled_label": shuffled_ce,
            },
        },
        "retention": {
            "replay_ce_before": replay_before,
            "replay_ce_after": replay_after,
            "replay_ce_change": replay_change,
            "replay_accuracy_before": replay_accuracy_before,
            "replay_accuracy_after": replay_accuracy_after,
            "tolerance": max_retention_loss,
            "passed": bool(replay_change is not None
                           and replay_change <= max_retention_loss),
        },
        "useful_learning": {
            "passed": bool(useful_passed),
            "criteria": criteria,
            "claim": "future-query transfer under a distinct query, not native capability",
            "parameter_change_alone_is_not_evidence": True,
            "future_ce_improvement": future_improvement,
            "wrong_world_margin": (
                None if future_after is None or wrong_world_ce is None
                else wrong_world_ce - future_after
            ),
            "shuffled_label_margin": (
                None if future_after is None or shuffled_ce is None
                else shuffled_ce - future_after
            ),
        },
        "cost": {
            "arm": arm,
            "per_update": {
                "declared_scalar_ops": update_costs[0]["declared_scalar_ops"],
                "measured_seconds_mean": float(
                    np.mean([item["measured_seconds"] for item in update_costs])
                ),
                "delta_norm_mean": float(
                    np.mean([item["delta_norm"] for item in update_costs])
                ),
                "changed_entries_mean": float(
                    np.mean([item["changed_entries"] for item in update_costs])
                ),
            },
            "declared_comparison": {
                ARM_SGD: declared_update_ops(ARM_SGD, backbone.shapes),
                ARM_DELTA: declared_update_ops(ARM_DELTA, backbone.shapes),
                ARM_ADAMW_REFERENCE: declared_update_ops(
                    ARM_ADAMW_REFERENCE, backbone.shapes
                ),
            },
            "declared_meta_comparison": {
                f"{ARM_SGD}/{DERIVATIVE_EXACT}": declared_meta_ops(
                    ARM_SGD, DERIVATIVE_EXACT, backbone.shapes
                ),
                f"{ARM_SGD}/{DERIVATIVE_TRUNCATED}": declared_meta_ops(
                    ARM_SGD, DERIVATIVE_TRUNCATED, backbone.shapes
                ),
                f"{ARM_DELTA}/{DERIVATIVE_EXACT}": declared_meta_ops(
                    ARM_DELTA, DERIVATIVE_EXACT, backbone.shapes
                ),
            },
            "measurement_scope": "cpu_fixture_not_native",
        },
        "rows": rows,
    }


def _global_norm(phi: WriterParams) -> float:
    squared = sum(float(np.sum(np.square(getattr(phi, name)))) for name in _ARRAY_NAMES)
    squared += float(phi.gain_log) ** 2
    return float(math.sqrt(max(squared, 0.0)))


def _apply_gradient(phi: WriterParams, grad: WriterParams, lr: float) -> WriterParams:
    arrays = phi.arrays()
    for name in _ARRAY_NAMES:
        arrays[name] = arrays[name] - lr * getattr(grad, name)
    gain_log = phi.gain_log - lr * float(grad.gain_log)
    return phi.with_arrays(gain_log=gain_log, **arrays)


def _nonzero_scalar_count(phi: WriterParams) -> int:
    count = sum(int(np.count_nonzero(getattr(phi, name))) for name in _ARRAY_NAMES)
    count += int(phi.gain_log != 0.0)
    return count


def _changed_scalars(before: WriterParams, after: WriterParams) -> int:
    changed = 0
    for name in _ARRAY_NAMES:
        changed += int(np.count_nonzero(getattr(before, name) != getattr(after, name)))
    changed += int(before.gain_log != after.gain_log)
    return changed


def _max_abs_change(before: WriterParams, after: WriterParams) -> float:
    values = [
        float(np.max(np.abs(getattr(before, name) - getattr(after, name))))
        for name in _ARRAY_NAMES
    ]
    values.append(abs(before.gain_log - after.gain_log))
    return max(values) if values else 0.0


def train_writer(
    backbone: FixedBackbone,
    episodes: Sequence[Episode],
    *,
    arm: str = ARM_SGD,
    initial_phi: WriterParams | None = None,
    outer_steps: int = 200,
    outer_lr: float = 0.05,
    inner_lr: float = 0.7,
    replay_weight: float = 0.5,
    cost_weight: float = 0.1,
    cost_reference: float = 0.5,
    derivative_mode: str = DERIVATIVE_EXACT,
    gradient_clip: float = 1.0,
    seed: int = 0,
    evaluation_episodes: Sequence[Episode] | None = None,
    min_gain: float = 1e-3,
    max_retention_loss: float = 0.05,
) -> dict:
    """Train offline writer machinery through the exact one-step update.

    This is offline ``phi`` training, not a native Spark run.  Each outer step
    consumes one episode's write query and future/replay queries and performs
    exactly one functional inner ``D`` update.  The returned report keeps
    execution, parameter updates, transfer, retention, useful learning and cost
    in separate sections.
    """
    _require(arm in FUNCTIONAL_ARMS,
             "train_writer supports functional_sgd/functional_delta only; "
             "the imperative AdamW reference is non-differentiable")
    _require(derivative_mode in DERIVATIVE_MODES,
             f"derivative_mode must be one of {DERIVATIVE_MODES}")
    if arm == ARM_DELTA:
        _require(derivative_mode == DERIVATIVE_EXACT,
                 "functional_delta has only an exact first-order derivative")
    _require(isinstance(outer_steps, int) and outer_steps >= 0,
             "outer_steps must be a nonnegative integer")
    _require(math.isfinite(outer_lr) and outer_lr > 0.0,
             "outer_lr must be positive")
    _require(math.isfinite(inner_lr) and inner_lr > 0.0,
             "inner_lr must be positive")
    _require(math.isfinite(replay_weight) and replay_weight >= 0.0,
             "replay_weight must be nonnegative")
    _require(math.isfinite(cost_weight) and cost_weight >= 0.0,
             "cost_weight must be nonnegative")
    _require(math.isfinite(cost_reference) and cost_reference > 0.0,
             "cost_reference must be positive")
    _require(math.isfinite(gradient_clip) and gradient_clip > 0.0,
             "gradient_clip must be positive")

    episodes = tuple(episodes)
    _require(bool(episodes), "at least one training episode is required")
    for episode in episodes:
        _require(isinstance(episode, Episode), "training entries must be Episodes")
        _require(episode.split != SEALED_SPLIT, "sealed episodes are closed")
    if evaluation_episodes is None:
        evaluation = episodes
        evaluation_scope = "training_episodes_audit"
    else:
        evaluation = tuple(evaluation_episodes)
        _require(bool(evaluation), "evaluation episodes must be nonempty")
        for episode in evaluation:
            _require(isinstance(episode, Episode), "evaluation entries must be Episodes")
            _require(episode.split != SEALED_SPLIT, "sealed episodes are closed")
        evaluation_scope = "held_out_development"

    phi_before = initial_phi if initial_phi is not None else WriterParams.initial(
        backbone.shapes, seed=seed
    )
    _require(isinstance(phi_before, WriterParams), "initial_phi must be WriterParams")
    _require(phi_before.shapes == backbone.shapes, "initial_phi/backbone shapes must match")
    phi = phi_before

    history: list[dict] = []
    completed = 0
    active_scalar_updates = 0
    status = "completed"
    error: str | None = None
    for index in range(outer_steps):
        episode = episodes[index % len(episodes)]
        try:
            result = outer_value_and_gradient(
                backbone, phi, episode,
                arm=arm,
                inner_lr=inner_lr,
                replay_weight=replay_weight,
                cost_weight=cost_weight,
                cost_reference=cost_reference,
                derivative_mode=derivative_mode,
            )
            grad_norm = result.grad.norm()
            _require(math.isfinite(result.value) and math.isfinite(grad_norm),
                     "non-finite outer objective or gradient")
            scale = min(1.0, gradient_clip / max(grad_norm, 1e-12))
            active_scalar_updates += _nonzero_scalar_count(result.grad)
            phi = _apply_gradient(phi, result.grad, outer_lr * scale)
            completed += 1
            history.append(
                {
                    "step": index + 1,
                    "episode_id": episode.episode_id,
                    "arm": arm,
                    "value": result.value,
                    "future_loss": result.future_loss,
                    "replay_loss": result.replay_loss,
                    "cost_norm": result.cost_norm,
                    "cost_term": result.cost_term,
                    "grad_norm": grad_norm,
                    "grad_scale": scale,
                    "gain": result.step.gain,
                    "delta_norm": result.step.cost.delta_norm,
                    "changed_entries": result.step.cost.changed_entries,
                    "measured_seconds": result.step.cost.measured_seconds,
                    "future_accuracy": result.future_accuracy,
                    "replay_accuracy": result.replay_accuracy,
                }
            )
        except (ValueError, FloatingPointError) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            break

    if status == "completed" and completed != outer_steps:
        status = "failed"
        error = "outer step count mismatch"

    evaluation_result = evaluate_writer(
        backbone, phi, evaluation,
        arm=arm,
        inner_lr=inner_lr,
        replay_weight=replay_weight,
        cost_weight=cost_weight,
        cost_reference=cost_reference,
        derivative_mode=derivative_mode,
        min_gain=min_gain,
        max_retention_loss=max_retention_loss,
    )

    grad_norms = [row["grad_norm"] for row in history]
    measured = [row["delta_norm"] for row in history]
    measured_seconds = [row["measured_seconds"] for row in history]
    per_step_ops = declared_update_ops(arm, backbone.shapes)
    training_cost = {
        "arm": arm,
        "outer_steps": completed,
        "inner_steps": completed,
        "declared_scalar_ops_per_update": per_step_ops,
        "declared_scalar_ops_total": int(per_step_ops * completed),
        "mean_grad_norm": _mean_or_none(grad_norms),
        "max_grad_norm": max(grad_norms) if grad_norms else None,
        "mean_delta_norm": _mean_or_none(measured),
        "max_delta_norm": max(measured) if measured else None,
        "mean_changed_entries": _mean_or_none(
            [row["changed_entries"] for row in history]
        ),
        "mean_update_wall_seconds": _mean_or_none(measured_seconds),
        "total_update_wall_seconds": float(sum(measured_seconds)),
        "measurement_scope": "cpu_fixture_not_native",
        "cost_term": {
            "cost_weight": cost_weight,
            "cost_reference": cost_reference,
            "note": "differentiable write-payload surrogate; declared compute and "
                    "wall-clock are reported separately",
        },
    }

    report = {
        "format": REPORT_FORMAT,
        "evidence_class": EVIDENCE_CLASS,
        "native_training": False,
        "execution": {
            "status": status,
            "error": error,
            "arm": arm,
            "derivative_mode": derivative_mode,
            "outer_steps_requested": outer_steps,
            "outer_steps_completed": completed,
            "inner_steps": completed,
            "training_episodes": len(episodes),
            "training_worlds": tuple(sorted({episode.world_id for episode in episodes})),
            "evaluation_episodes": len(evaluation),
            "evaluation_worlds": tuple(
                sorted({episode.world_id for episode in evaluation})
            ),
            "evaluation_scope": evaluation_scope,
            "sealed_opened": False,
            "native_capability_claim": False,
            "one_learning_clock": True,
            "rsi": False,
            "multi_clock_promotion": False,
        },
        "parameter_updates": {
            "online_d_updates": completed,
            "online_d_parameter_count": (
                backbone.shapes.latent * backbone.shapes.hidden
            ),
            "online_d_mean_changed_entries": training_cost["mean_changed_entries"],
            "online_d_mean_delta_norm": training_cost["mean_delta_norm"],
            "online_d_max_delta_norm": training_cost["max_delta_norm"],
            "online_d_note": (
                "one functional D update per episode from the fixed D0; the "
                "fixture does not pretend the per-episode D-prime states are one "
                "persistent final D"
            ),
            "outer_optimizer_steps": completed,
            "scalar_parameter_count": phi.parameter_count,
            "total_scalar_updates": int(active_scalar_updates),
            "total_scalar_slots_visited": int(phi.parameter_count * completed),
            "mean_active_scalars_per_step": (
                int(active_scalar_updates // completed) if completed else 0
            ),
            "changed_scalars": _changed_scalars(phi_before, phi),
            "counting": (
                "total_scalar_updates counts nonzero gradient entries applied per "
                "outer step; total_scalar_slots_visited counts every phi scalar "
                "slot per outer step; changed_scalars counts unique final changes"
            ),
            "phi_norm_before": phi_before.norm(),
            "phi_norm_after": phi.norm(),
            "max_abs_parameter_change": _max_abs_change(phi_before, phi),
            "phi_digest_before": phi_before.digest(),
            "phi_digest_after": phi.digest(),
            "note": "parameter changes alone do not establish useful learning",
        },
        "derivatives": {
            "mode": derivative_mode,
            "exact_meta_gradient": bool(
                (arm == ARM_SGD and derivative_mode == DERIVATIVE_EXACT)
                or arm == ARM_DELTA
            ),
            "second_order_softmax_hessian": bool(
                arm == ARM_SGD and derivative_mode == DERIVATIVE_EXACT
            ),
            "first_order_truncated": bool(
                arm == ARM_SGD and derivative_mode == DERIVATIVE_TRUNCATED
            ),
            "delta_first_order_exact": bool(arm == ARM_DELTA),
            "imperative_adamw_differentiable": False,
            "inner_equation": (
                "D' = D - inner_lr * gain * R^T (p_t - q_t) h(z_t)^T"
                if arm == ARM_SGD
                else "D' = D + inner_lr * gain * v h(x_t)^T"
            ),
            "outer_equation": (
                "J = CE_future(D') + replay_weight * CE_replay(D') "
                "+ cost_weight * ||D'-D||_F / cost_reference"
            ),
            "numerical_check_api": "finite_difference_gradient_check",
            "scope": "cpu_engineering_fixture_not_native_capability",
        },
        "transfer": evaluation_result["transfer"],
        "retention": evaluation_result["retention"],
        "useful_learning": evaluation_result["useful_learning"],
        "cost": {
            **evaluation_result["cost"],
            "training": training_cost,
        },
        "clock": {
            "online_learning_clock": "single_d_update_per_episode",
            "online_d_updates": completed,
            "inner_steps_per_online_update": 1,
            "offline_phi_outer_steps": completed,
            "multi_clock": False,
            "rsi": False,
            "promotion": False,
            "note": "offline phi training is pre-deployment; it is not a second "
                    "online learning clock",
        },
        "sealed": {
            "opened": False,
            "evaluated": False,
            "access": "closed",
            "reason": "sealed evaluation is not an outer-training query set",
        },
        "limits": list(_LIMITS),
        "history": history[-50:],
    }
    return report


# ---------------------------------------------------------------------------
# Public inventory
# ---------------------------------------------------------------------------

PUBLIC_APIS = (
    "WriterShapes",
    "FixedBackbone",
    "WriterParams",
    "Example",
    "Episode",
    "WorldSuite",
    "SealedEvaluationClosed",
    "make_world_suite",
    "functional_sgd_step",
    "functional_delta_step",
    "sgd_meta_gradient",
    "delta_meta_gradient",
    "outer_value_and_gradient",
    "finite_difference_gradient_check",
    "evaluate_writer",
    "train_writer",
    "declared_update_ops",
    "declared_meta_ops",
    "adamw_reference_step",
    "softmax",
    "one_hot",
    "cross_entropy",
)
