"""Verified-learning contract: taught-rule transfer, not demonstration recall.

This module is CPU procedure plumbing.  It defines an explicit rule-learning
objective, bounded adapter updates, parameter-matched core controls, detached
audit heads, exact resource accounting and two learning-policy arms with
identical observations/update counts/limits.  It does *not* demonstrate
verified learning, general improvement or recursive self-improvement; those
claims require the separately gated native experiment.

The objective is response cross-entropy through a frozen base decoder on
demonstrations only.  Held-out transfer targets are scored after adaptation and
never enter a gradient.  A demonstration-recall control can score 100% on the
demonstrations while scoring zero on unseen inputs; the gate therefore requires
a held-out transfer gain over no-update and fake-update controls.

``zero``/``one``/``two`` recurrence steps and the parameter-matched
feed-forward/no-core arms are controls, not capability claims.  The audit head
is detached from adapter gradients and its logits are rejected by the response
objective, so a monitor cannot substitute for the decoder objective or carry an
answer.  Personality fidelity is a separate synthetic style trace; it is never
combined with factual or memory accuracy, and authentic personality content
remains blocked on a permitted user specification.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .architecture.contracts import CoreSpec, fingerprint
from .architecture.core import (
    CognitiveCore,
    FeedForwardCore,
    cognitive_parameter_count,
    core_parameter_count,
)
from .architecture.heads import TaskSuccessHead

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROTOCOL_FORMAT = 'replica.verified_learning.protocol.v1'
CONFIG_FORMAT = 'replica.verified_learning.config.v1'
PREPARED_FORMAT = 'replica.verified_learning.prepared.v1'
FIXTURE_FORMAT = 'replica.verified_learning.fixture_report.v1'
POLICY_FORMAT = 'replica.verified_learning.policy_comparison.v1'
ACCOUNTING_FORMAT = 'replica.verified_learning.accounting.v1'

SPLITS = ('train', 'dev', 'heldout')
OBSERVATION_KINDS = ('demo', 'transfer', 'composition')
CORE_ARMS = ('no_core', 'feedforward', 'recurrent_0', 'recurrent_1',
             'recurrent_2')
REQUIRED_CONTROL_ARMS = ('no_core', 'feedforward', 'recurrent_0',
                         'recurrent_1', 'recurrent_2')
POLICY_ARMS = ('frozen_policy', 'updated_policy')

TOKEN_PAD, TOKEN_BOS, TOKEN_RULE, TOKEN_X, TOKEN_ARROW = 0, 1, 2, 3, 4
TOKEN_BASE = 5
DIGITS = 10
DEFAULT_SEQUENCE = 7

COUNTER_KEYS = (
    'inference_batches', 'inference_examples', 'inference_positions',
    'forward_batches', 'forward_examples', 'forward_positions',
    'backward_batches', 'optimizer_steps', 'gradient_clip_calls',
)

IMPLEMENTATION_PATHS = (
    'src/replica_lab/verified_learning.py',
    'scripts/run_verified_learning.py',
    'configs/verified_learning_v1.json',
    'tests/test_verified_learning.py',
)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(Path(path).read_bytes())


def _sha_json(value) -> str:
    return fingerprint(value)


def _write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + '\n')


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Tensor, name: str) -> None:
    _require(value.is_floating_point(), f'{name} must be floating point')
    _require(bool(torch.isfinite(value).all()), f'{name} must be finite')


@contextmanager
def _seeded(seed: int):
    """Deterministic module construction without leaking global RNG state."""
    _require(type(seed) is int and seed >= 0, 'seed must be a nonnegative integer')
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


# ---------------------------------------------------------------------------
# Deterministic synthetic worlds and the explicit transfer distinction


@dataclass(frozen=True)
class RuleWorld:
    """One affine nonce rule ``y = (a*x + b) mod modulus`` over owned inputs.

    ``demo_inputs`` are the only training targets.  ``transfer_inputs`` are
    disjoint held-out applications, and ``composition_inputs`` require applying
    the rule twice.  No target is stored here twice; targets are derived.
    """
    world_id: str
    split: str
    family: str
    modulus: int
    a: int
    b: int
    demo_inputs: tuple[int, ...]
    transfer_inputs: tuple[int, ...]
    composition_inputs: tuple[int, ...]
    style: tuple[int, ...]

    def validate(self) -> 'RuleWorld':
        _require(self.split in SPLITS, 'unknown split')
        _require(isinstance(self.world_id, str) and self.world_id, 'world id required')
        _require(isinstance(self.family, str) and self.family, 'family required')
        _require(type(self.modulus) is int and 2 <= self.modulus <= DIGITS,
                 'modulus must be in [2, 10]')
        _require(type(self.a) is int and 0 < self.a < self.modulus,
                 'rule slope must be a nonzero residue')
        _require(type(self.b) is int and 0 <= self.b < self.modulus,
                 'rule offset must be a residue')
        for name in ('demo_inputs', 'transfer_inputs', 'composition_inputs'):
            values = getattr(self, name)
            _require(isinstance(values, tuple) and values, f'{name} required')
            _require(all(type(v) is int and 0 <= v < self.modulus for v in values),
                     f'{name} must be residues')
            _require(len(set(values)) == len(values), f'{name} must be distinct')
        _require(not set(self.demo_inputs) & set(self.transfer_inputs),
                 'held-out transfer inputs overlap demonstrations')
        _require(not set(self.demo_inputs) & set(self.composition_inputs),
                 'composition inputs overlap demonstrations')
        _require(isinstance(self.style, tuple) and self.style
                 and all(type(v) is int and v >= 0 for v in self.style),
                 'synthetic style trace must be nonnegative integers')
        return self

    def target(self, x: int) -> int:
        return (self.a * x + self.b) % self.modulus

    def rows(self, kind: str) -> list[dict]:
        _require(kind in OBSERVATION_KINDS, 'unknown observation kind')
        if kind == 'demo':
            inputs, source = self.demo_inputs, 'demonstration'
            targets = [self.target(x) for x in inputs]
        elif kind == 'transfer':
            inputs, source = self.transfer_inputs, 'heldout_rule'
            targets = [self.target(x) for x in inputs]
        else:
            inputs, source = self.composition_inputs, 'composition'
            targets = [self.target(self.target(x)) for x in inputs]
        return [{'world_id': self.world_id, 'kind': kind, 'source': source,
                 'input': x, 'target': y}
                for x, y in zip(inputs, targets)]

    def to_dict(self) -> dict:
        return {
            'world_id': self.world_id, 'split': self.split,
            'family': self.family, 'modulus': self.modulus, 'a': self.a,
            'b': self.b, 'demo_inputs': list(self.demo_inputs),
            'transfer_inputs': list(self.transfer_inputs),
            'composition_inputs': list(self.composition_inputs),
            'style': list(self.style),
        }

    @classmethod
    def from_dict(cls, value) -> 'RuleWorld':
        _require(isinstance(value, dict), 'world must be an object')
        try:
            world = cls(
                world_id=value['world_id'], split=value['split'],
                family=value['family'], modulus=value['modulus'],
                a=value['a'], b=value['b'],
                demo_inputs=tuple(value['demo_inputs']),
                transfer_inputs=tuple(value['transfer_inputs']),
                composition_inputs=tuple(value['composition_inputs']),
                style=tuple(value['style']))
        except (KeyError, TypeError) as error:
            raise ValueError(f'malformed world record: {error}') from error
        return world.validate()


DEFAULT_FAMILIES_PER_SPLIT = {'train': 4, 'dev': 3, 'heldout': 4}


def generate_rule_worlds(*, seed: int = 20261025, modulus: int = 10,
                         families_per_split: dict | None = None,
                         demo_count: int = 4, composition_count: int = 2,
                         style_width: int = 4,
                         style_values: int = 4) -> list[RuleWorld]:
    """Generate disjoint rule families and input partitions deterministically."""
    _require(type(seed) is int and seed >= 0, 'seed must be a nonnegative integer')
    _require(type(modulus) is int and 2 <= modulus <= DIGITS,
             'modulus must be in [2, 10]')
    counts = dict(DEFAULT_FAMILIES_PER_SPLIT if families_per_split is None
                  else families_per_split)
    _require(set(counts) == set(SPLITS), 'family counts must name each split')
    _require(all(type(v) is int and v >= 1 for v in counts.values()),
             'positive family counts required')
    _require(type(demo_count) is int and demo_count >= 1, 'demo_count required')
    _require(type(composition_count) is int and composition_count >= 1,
             'composition_count required')
    required_inputs = 2 * demo_count + composition_count
    _require(required_inputs <= modulus,
             'modulus cannot supply disjoint demo/transfer/composition inputs')
    _require(type(style_width) is int and style_width >= 1, 'style width required')
    _require(type(style_values) is int and style_values >= 1,
             'style values required')
    candidates = [(a, b) for a in range(1, modulus) for b in range(modulus)]
    total = sum(counts.values())
    _require(total <= len(candidates), 'not enough distinct rule families')

    rng = random.Random(seed)
    rng.shuffle(candidates)
    worlds = []
    cursor = 0
    for split in SPLITS:
        for index in range(counts[split]):
            a, b = candidates[cursor]
            cursor += 1
            inputs = list(range(modulus))
            rng.shuffle(inputs)
            demos = tuple(inputs[:demo_count])
            transfer = tuple(inputs[demo_count:2 * demo_count])
            composition = tuple(inputs[2 * demo_count:2 * demo_count
                                         + composition_count])
            style = tuple(rng.randrange(style_values) for _ in range(style_width))
            world = RuleWorld(
                world_id=f'{split}-{index:02d}-a{a}-b{b}-m{modulus}',
                split=split, family=f'rule_a{a}_b{b}_m{modulus}',
                modulus=modulus, a=a, b=b, demo_inputs=demos,
                transfer_inputs=transfer, composition_inputs=composition,
                style=style).validate()
            worlds.append(world)
    families = [world.family for world in worlds]
    _require(len(families) == len(set(families)), 'rule families must be disjoint')
    return worlds


def observation_digest(worlds) -> str:
    """Digest every observation and held-out input, but never a hidden label."""
    payload = []
    for world in worlds:
        payload.append({
            'world_id': world.world_id, 'split': world.split,
            'family': world.family, 'rule': [world.a, world.b, world.modulus],
            'demo': [{'input': x, 'target': world.target(x)}
                     for x in world.demo_inputs],
            'transfer_inputs': list(world.transfer_inputs),
            'composition_inputs': list(world.composition_inputs),
        })
    return _sha_json(payload)


# ---------------------------------------------------------------------------
# Model: sensors -> core control -> reader -> frozen base decoder


@dataclass(frozen=True)
class ResponseLogits:
    """Answer logits produced by the frozen base decoder."""
    logits: Tensor
    producer: str = 'base_decoder'
    decoder_frozen: bool = True

    def __post_init__(self):
        _require(isinstance(self.logits, Tensor) and self.logits.ndim == 2,
                 'response logits must be [batch, vocabulary]')
        _finite(self.logits, 'response logits')
        _require(self.producer == 'base_decoder',
                 'response logits must come from the base decoder')
        _require(self.decoder_frozen is True,
                 'response objective requires a frozen base decoder')


@dataclass(frozen=True)
class AuditLogits:
    """Detached monitor logits; never an answer, reward or objective input."""
    logits: Tensor
    producer: str = 'detached_audit'

    def __post_init__(self):
        _require(isinstance(self.logits, Tensor) and self.logits.ndim == 1,
                 'audit logits must be [batch]')
        _finite(self.logits, 'audit logits')
        _require(self.producer == 'detached_audit',
                 'audit logits must carry the audit role')


def response_cross_entropy(output, targets: Tensor) -> Tensor:
    """Response CE through the base decoder; audit logits are rejected."""
    if isinstance(output, AuditLogits):
        raise TypeError('audit logits cannot be the response objective')
    if not isinstance(output, ResponseLogits):
        raise TypeError('response objective requires base-decoder ResponseLogits')
    _require(targets.dtype == torch.long and targets.ndim == 1,
             'targets must be a [batch] long tensor')
    _require(output.logits.shape[0] == targets.numel(),
             'logits and targets must have the same batch size')
    return F.cross_entropy(output.logits, targets)


def audit_cross_entropy(output: AuditLogits, targets: Tensor) -> Tensor:
    """Separate supervised objective for the detached monitor only."""
    if not isinstance(output, AuditLogits):
        raise TypeError('audit objective requires detached AuditLogits')
    _require(targets.dtype == torch.float32 and targets.shape == output.logits.shape,
             'audit targets must match [batch] logits')
    return F.binary_cross_entropy_with_logits(output.logits, targets)


class TokenSensor(nn.Module):
    """Learned token/position sensors; no pretrained tokenizer is involved."""

    def __init__(self, vocabulary: int, width: int, max_positions: int) -> None:
        super().__init__()
        _require(vocabulary > TOKEN_BASE and width > 0 and max_positions > 0,
                 'positive sensor dimensions required')
        self.vocabulary, self.width, self.max_positions = (
            vocabulary, width, max_positions)
        self.embedding = nn.Embedding(vocabulary, width)
        self.position = nn.Embedding(max_positions, width)
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, width)

    def forward(self, token_ids: Tensor) -> Tensor:
        _require(token_ids.dtype == torch.long and token_ids.ndim == 2,
                 'token ids must be [batch, positions] long')
        _require(token_ids.shape[1] <= self.max_positions,
                 'token sequence exceeds declared maximum')
        _require(bool(((token_ids >= 0) & (token_ids < self.vocabulary)).all()),
                 'token id outside vocabulary')
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        values = self.embedding(token_ids) + self.position(positions)[None]
        return self.projection(self.norm(values))


class WorkspaceReader(nn.Module):
    """Read pooled workspace and the current token into decoder width."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(2 * width, width)
        self.down = nn.Linear(width, width)

    def forward(self, workspace: Tensor, sense: Tensor) -> Tensor:
        _require(workspace.ndim == 3 and sense.ndim == 3,
                 'workspace and sense must be rank three')
        _require(workspace.shape[0] == sense.shape[0]
                 and workspace.shape[-1] == sense.shape[-1],
                 'workspace and sense must share batch/width')
        pooled = workspace.mean(dim=1)
        features = torch.cat((self.norm(pooled), self.norm(sense[:, -1])), dim=-1)
        return self.down(F.gelu(self.up(features)))


class FrozenResponseDecoder(nn.Module):
    """Frozen base decoder; gradients pass through it to sensors/core/readers."""

    def __init__(self, width: int, vocabulary: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, vocabulary, bias=False)
        for parameter in self.projection.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.projection(hidden)


class RuleResponseModel(nn.Module):
    """Tiny adapter around one matched core control and a frozen decoder.

    No storage lives in forward.  The audit head receives only detached task
    features/workspaces, and is excluded from the response objective.
    """

    def __init__(self, spec: CoreSpec, *, core_impl: str = 'recurrent',
                 steps: int = 2, vocabulary: int = TOKEN_BASE + DIGITS,
                 max_positions: int = DEFAULT_SEQUENCE,
                 audit_hidden: int = 16) -> None:
        super().__init__()
        _require(core_impl in ('none', 'feedforward', 'recurrent'),
                 'unknown core implementation')
        _require(type(steps) is int and 0 <= steps <= 2,
                 'verified-learning controls bound recurrence to 0/1/2 steps')
        _require(vocabulary > TOKEN_BASE and max_positions > 0,
                 'positive vocabulary/max positions required')
        self.spec, self.core_impl, self.steps = spec, core_impl, steps
        self.sensor = TokenSensor(vocabulary, spec.width, max_positions)
        if core_impl == 'recurrent':
            self.core = CognitiveCore(spec)
        elif core_impl == 'feedforward':
            _require(steps == 0, 'feed-forward control has no recurrence steps')
            self.core = FeedForwardCore(spec)
        else:
            _require(steps == 0, 'no-core control has no recurrence steps')
            self.core = None
        self.reader = WorkspaceReader(spec.width)
        self.decoder = FrozenResponseDecoder(spec.width, vocabulary)
        self.audit = TaskSuccessHead(spec, hidden=audit_hidden)

    def _workspace(self, sense: Tensor, task_features: Tensor) -> Tensor:
        batch, positions, width = sense.shape
        _require(batch == 1,
                 'core controls own one task boundary; encode one example at a time')
        valid = torch.ones(1, positions, dtype=torch.bool, device=sense.device)
        if self.core_impl == 'none':
            return sense.mean(dim=1, keepdim=True).expand(-1, self.spec.slots, -1)
        if self.core_impl == 'feedforward':
            empty = torch.zeros(1, 1, width, device=sense.device,
                                dtype=sense.dtype)
            workspace, _ = self.core(sense.mean(dim=1), empty.mean(dim=1),
                                     task_features)
            return workspace
        memory = torch.zeros(1, 1, width, device=sense.device, dtype=sense.dtype)
        memory_valid = torch.zeros(1, 1, dtype=torch.bool, device=sense.device)
        workspace, _ = self.core(sense, valid, memory, memory_valid,
                                 task_features, steps=self.steps)
        return workspace

    def forward(self, token_ids: Tensor, task_features: Tensor) -> ResponseLogits:
        _require(token_ids.ndim == 2 and task_features.ndim == 2,
                 'token ids and task features must be rank two')
        _require(token_ids.shape[0] == task_features.shape[0],
                 'token ids and task features must share a batch')
        _require(task_features.shape[1] == self.spec.task_features,
                 'task features do not match the core spec')
        _finite(task_features, 'task features')
        logits = []
        for index in range(token_ids.shape[0]):
            sense = self.sensor(token_ids[index:index + 1])
            workspace = self._workspace(sense, task_features[index:index + 1])
            hidden = self.reader(workspace, sense)
            logits.append(self.decoder(hidden))
        return ResponseLogits(torch.cat(logits, dim=0))

    def audit_logits(self, token_ids: Tensor,
                     task_features: Tensor) -> AuditLogits:
        """Detached monitor output; no target, no answer and no adapter gradient."""
        _require(token_ids.ndim == 2 and task_features.ndim == 2,
                 'token ids and task features must be rank two')
        values = []
        for index in range(token_ids.shape[0]):
            sense = self.sensor(token_ids[index:index + 1])
            workspace = self._workspace(sense, task_features[index:index + 1])
            values.append(self.audit(task_features[index:index + 1],
                                     workspace))
        return AuditLogits(torch.cat(values, dim=0))


# ---------------------------------------------------------------------------
# Deterministic encoding, scope and resource accounting


def encode_example(world: RuleWorld, x: int) -> tuple[list[int], list[float]]:
    """Prompt tokens and prompt-only task features; no target is encoded."""
    _require(0 <= x < world.modulus, 'input outside the rule domain')
    tokens = [TOKEN_BOS, TOKEN_RULE, TOKEN_BASE + world.a,
              TOKEN_BASE + world.b, TOKEN_X, TOKEN_BASE + x, TOKEN_ARROW]
    features = [world.a / world.modulus, world.b / world.modulus,
                world.modulus / DIGITS, 1.0]
    return tokens, features


def encode_rows(worlds, kind: str) -> dict:
    """Encode one observation kind without ever reading a hidden label."""
    _require(kind in OBSERVATION_KINDS, 'unknown observation kind')
    token_rows, feature_rows, targets, world_ids, inputs = [], [], [], [], []
    for world in worlds:
        for row in world.rows(kind):
            tokens, features = encode_example(world, row['input'])
            token_rows.append(tokens)
            feature_rows.append(features)
            targets.append(row['target'])
            world_ids.append(row['world_id'])
            inputs.append(row['input'])
    _require(token_rows, 'no rows to encode')
    _require(len({len(row) for row in token_rows}) == 1,
             'all encoded prompts must share a length')
    return {
        'token_ids': torch.tensor(token_rows, dtype=torch.long),
        'task_features': torch.tensor(feature_rows, dtype=torch.float32),
        'targets': torch.tensor(targets, dtype=torch.long),
        'world_ids': tuple(world_ids),
        'inputs': tuple(inputs),
        'kind': kind,
        'sequence_positions': len(token_rows[0]),
    }


def example_digest(encoded: dict) -> str:
    return _sha_json({
        'kind': encoded['kind'],
        'token_ids': encoded['token_ids'].tolist(),
        'task_features': [[round(v, 8) for v in row]
                          for row in encoded['task_features'].tolist()],
        'targets': encoded['targets'].tolist(),
        'world_ids': list(encoded['world_ids']),
    })


@dataclass(frozen=True)
class ResourceLimits:
    max_forward_batches: int = 4096
    max_forward_examples: int = 65536
    max_forward_positions: int = 524288
    max_backward_batches: int = 4096
    max_inference_batches: int = 65536
    max_inference_examples: int = 262144
    max_inference_positions: int = 2097152
    max_optimizer_steps: int = 1024
    max_gradient_clip_calls: int = 4096
    max_adapter_parameters: int = 250000
    max_optimizer_state_elements: int = 0
    max_parameter_delta_norm: float = 100.0

    def validate(self) -> 'ResourceLimits':
        for name in ('max_forward_batches', 'max_forward_examples',
                     'max_forward_positions', 'max_backward_batches',
                     'max_inference_batches', 'max_inference_examples',
                     'max_inference_positions', 'max_optimizer_steps',
                     'max_gradient_clip_calls', 'max_adapter_parameters',
                     'max_optimizer_state_elements'):
            value = getattr(self, name)
            _require(type(value) is int and value >= 0,
                     f'{name} must be a nonnegative integer')
        _require(isinstance(self.max_parameter_delta_norm, (int, float))
                 and math.isfinite(self.max_parameter_delta_norm)
                 and self.max_parameter_delta_norm >= 0,
                 'max_parameter_delta_norm must be finite and nonnegative')
        return self

    def to_dict(self) -> dict:
        return {
            'max_forward_batches': self.max_forward_batches,
            'max_forward_examples': self.max_forward_examples,
            'max_forward_positions': self.max_forward_positions,
            'max_backward_batches': self.max_backward_batches,
            'max_inference_batches': self.max_inference_batches,
            'max_inference_examples': self.max_inference_examples,
            'max_inference_positions': self.max_inference_positions,
            'max_optimizer_steps': self.max_optimizer_steps,
            'max_gradient_clip_calls': self.max_gradient_clip_calls,
            'max_adapter_parameters': self.max_adapter_parameters,
            'max_optimizer_state_elements': self.max_optimizer_state_elements,
            'max_parameter_delta_norm': self.max_parameter_delta_norm,
        }

    @classmethod
    def from_dict(cls, value) -> 'ResourceLimits':
        _require(isinstance(value, dict), 'resource limits must be an object')
        names = set(cls.__dataclass_fields__)
        _require(set(value) == names,
                 f'resource limit fields differ: missing={names-set(value)}, '
                 f'extra={set(value)-names}')
        return cls(**value).validate()


class ResourceBudgetExceeded(RuntimeError):
    """Raised before work would exceed an exact declared resource limit."""


class ResourceLedger:
    """Exact integer accounting; float parameter movement is an outcome field."""

    def __init__(self, limits: ResourceLimits, *, trainable_parameters: int = 0):
        self.limits = limits.validate()
        _require(type(trainable_parameters) is int and trainable_parameters >= 0,
                 'trainable parameter count must be nonnegative')
        _require(trainable_parameters <= limits.max_adapter_parameters,
                 'adapter parameter budget exceeded')
        self.counters = {key: 0 for key in COUNTER_KEYS}
        self.trainable_parameters = trainable_parameters
        self.optimizer_state_tensors = 0
        self.optimizer_state_elements = 0
        self.parameter_delta_norm = 0.0
        self.update_norms: list[float] = []

    def add(self, **increments) -> None:
        _require(set(increments) <= set(COUNTER_KEYS),
                 'unknown resource counter')
        for key, increment in increments.items():
            _require(type(increment) is int and increment >= 0,
                     'counter increments must be nonnegative integers')
            limit = self.limits.__dict__.get('max_' + key)
            if limit is None:
                raise ValueError(f'no declared limit for {key}')
            if self.counters[key] + increment > limit:
                raise ResourceBudgetExceeded(
                    f'{key} would exceed declared limit {limit}')
            self.counters[key] += increment

    def record_inference(self, batch: int, positions: int) -> None:
        self.add(inference_batches=1, inference_examples=batch,
                 inference_positions=batch * positions)

    def record_train_step(self, batch: int, positions: int) -> None:
        self.add(forward_batches=1, forward_examples=batch,
                 forward_positions=batch * positions)

    def record_backward(self) -> None:
        self.add(backward_batches=1)

    def record_optimizer_step(self) -> None:
        self.add(optimizer_steps=1)

    def record_gradient_clip(self) -> None:
        self.add(gradient_clip_calls=1)

    def finalize(self, optimizer, initial_state: dict, model: nn.Module) -> None:
        state = optimizer.state_dict().get('state', {})
        tensors = 0
        elements = 0
        for entry in state.values():
            for value in entry.values():
                if isinstance(value, Tensor):
                    tensors += 1
                    elements += value.numel()
        _require(elements <= self.limits.max_optimizer_state_elements,
                 'optimizer state budget exceeded')
        self.optimizer_state_tensors = tensors
        self.optimizer_state_elements = elements
        deltas = []
        for name, parameter in _trainable_parameters(model):
            delta = (parameter.detach() - initial_state[name]).float()
            deltas.append(delta.flatten())
        norm = (torch.cat(deltas).norm().item() if deltas else 0.0)
        _require(math.isfinite(norm), 'parameter delta norm must be finite')
        _require(norm <= self.limits.max_parameter_delta_norm,
                 'parameter delta norm exceeds declared bound')
        self.parameter_delta_norm = norm

    def as_dict(self) -> dict:
        return {
            'format': ACCOUNTING_FORMAT,
            'limits': self.limits.to_dict(),
            'counters': dict(self.counters),
            'trainable_parameters': self.trainable_parameters,
            'optimizer_state_tensors': self.optimizer_state_tensors,
            'optimizer_state_elements': self.optimizer_state_elements,
            'parameter_delta_norm': self.parameter_delta_norm,
            'update_norms': list(self.update_norms),
        }


def _trainable_parameters(model: nn.Module):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad]


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def configure_adapter_scope(model: RuleResponseModel, *,
                            include_audit: bool = False) -> dict:
    """Freeze the decoder and monitor; train sensors, core control and reader."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.sensor, model.reader):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if model.core is not None:
        for parameter in model.core.parameters():
            parameter.requires_grad_(True)
    if include_audit:
        for parameter in model.audit.parameters():
            parameter.requires_grad_(True)
    _require(all(not parameter.requires_grad
                 for parameter in model.decoder.parameters()),
             'base decoder must stay frozen')
    names = [name for name, _ in _trainable_parameters(model)]
    return {'trainable_names': names,
            'trainable_parameters': sum(
                parameter.numel() for _, parameter in _trainable_parameters(model)),
            'decoder_parameters': parameter_count(model.decoder),
            'audit_parameters': parameter_count(model.audit),
            'audit_trainable': any(parameter.requires_grad
                                   for parameter in model.audit.parameters())}


def parameter_state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().to(torch.float32).cpu().contiguous()
                      .numpy().tobytes())
    return digest.hexdigest()


def _snapshot(model: nn.Module) -> dict:
    return {name: parameter.detach().clone()
            for name, parameter in model.named_parameters()}


# ---------------------------------------------------------------------------
# Bounded adaptation on demonstrations only


def _batch_targets(encoded: dict, fake: bool, seed: int) -> Tensor:
    if not fake:
        return encoded['targets']
    generator = torch.Generator().manual_seed(seed)
    targets = encoded['targets']
    if targets.numel() > 1:
        permutation = torch.randperm(targets.numel(), generator=generator)
        shifted = targets[permutation]
        if bool(torch.equal(shifted, targets)):
            shifted = torch.roll(targets, shifts=1, dims=0)
        return shifted
    return (targets + 1) % DIGITS


def adapt_adapter(model: RuleResponseModel, demonstrations: dict, *,
                  learning_rate: float, update_steps: int,
                  limits: ResourceLimits, seed: int = 0,
                  max_grad_norm: float = 1.0,
                  fake_targets: bool = False) -> tuple[ResourceLedger, dict]:
    """Bounded full-batch updates on demonstrations; never on transfer labels."""
    _require(demonstrations.get('kind') == 'demo',
             'adaptation may use demonstrations only')
    _require(isinstance(learning_rate, (int, float)) and learning_rate > 0,
             'positive learning rate required')
    _require(type(update_steps) is int and update_steps >= 0,
             'update steps must be a nonnegative integer')
    _require(isinstance(max_grad_norm, (int, float)) and max_grad_norm > 0,
             'max_grad_norm must be positive')
    scope = configure_adapter_scope(model)
    parameters = [parameter for _, parameter in _trainable_parameters(model)]
    optimizer = torch.optim.SGD(parameters, lr=float(learning_rate),
                                momentum=0.0)
    ledger = ResourceLedger(limits, trainable_parameters=scope['trainable_parameters'])
    initial = _snapshot(model)
    targets = _batch_targets(demonstrations, fake_targets, seed)
    batch = demonstrations['token_ids'].shape[0]
    positions = demonstrations['sequence_positions']
    initial_loss = None
    previous = _snapshot(model)
    for step in range(update_steps):
        ledger.record_train_step(batch, positions)
        optimizer.zero_grad(set_to_none=True)
        logits = model(demonstrations['token_ids'],
                       demonstrations['task_features'])
        loss = response_cross_entropy(logits, targets)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        ledger.record_backward()
        loss.backward()
        clipped = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        ledger.record_gradient_clip()
        _require(math.isfinite(float(clipped)), 'gradient norm must be finite')
        optimizer.step()
        ledger.record_optimizer_step()
        deltas = [((parameter.detach() - previous[name]).float().flatten())
                  for name, parameter in _trainable_parameters(model)]
        ledger.update_norms.append(
            float(torch.cat(deltas).norm()) if deltas else 0.0)
        previous = _snapshot(model)
    ledger.finalize(optimizer, initial, model)
    if update_steps:
        ledger.record_inference(batch, positions)
        with torch.no_grad():
            final_loss = float(response_cross_entropy(
                model(demonstrations['token_ids'],
                      demonstrations['task_features']), targets).detach())
    else:
        final_loss = None
    summary = {
        'objective': 'response_cross_entropy_through_frozen_base_decoder',
        'training_kind': 'demonstrations_only',
        'transfer_targets_used_in_training': False,
        'fake_targets': bool(fake_targets),
        'learning_rate': float(learning_rate),
        'update_steps': update_steps,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'trainable_parameters': scope['trainable_parameters'],
        'audit_trainable': scope['audit_trainable'],
        'observations_sha256': example_digest(demonstrations),
        'accounting': ledger.as_dict(),
    }
    return ledger, summary


# ---------------------------------------------------------------------------
# Controls and scores


def build_model(spec: CoreSpec, *, core_impl: str, steps: int,
                seed: int = 0, audit_hidden: int = 16,
                max_positions: int = DEFAULT_SEQUENCE) -> RuleResponseModel:
    with _seeded(seed):
        return RuleResponseModel(spec, core_impl=core_impl, steps=steps,
                                 max_positions=max_positions,
                                 audit_hidden=audit_hidden)


def build_control_plan(spec: CoreSpec) -> dict:
    """Instantiate every control and record exact/relative parameter counts."""
    with _seeded(0):
        recurrent = CognitiveCore(spec)
        feedforward = FeedForwardCore(spec)
    recurrent_count = core_parameter_count(recurrent)
    for steps in (0, 1, 2):
        control = CognitiveCore(CoreSpec(**{**spec.__dict__, 'steps': steps}))
        _require(core_parameter_count(control) == recurrent_count,
                 'recurrence control parameters changed with step count')
    control_counts = {
        'no_core': 0,
        'feedforward': core_parameter_count(feedforward),
        'recurrent_0': recurrent_count,
        'recurrent_1': recurrent_count,
        'recurrent_2': recurrent_count,
    }
    _require(control_counts['recurrent_0'] == control_counts['recurrent_1']
             == control_counts['recurrent_2'],
             'recurrence controls must be parameter matched')
    relative = (control_counts['feedforward'] - recurrent_count) / recurrent_count
    _require(abs(relative) <= 0.05,
             'feed-forward control is not parameter matched within 5%')
    return {
        'arms': list(REQUIRED_CONTROL_ARMS),
        'parameter_counts': control_counts,
        'recurrent_parameters': recurrent_count,
        'feedforward_relative_delta': relative,
        'feedforward_matched_within_5_percent': True,
        'recurrence_steps': [0, 1, 2],
        'recurrence_parameter_matched': True,
        'analytic_recurrent_parameters': cognitive_parameter_count(spec),
        'spec': dict(spec.__dict__),
    }


def score_model(model: RuleResponseModel, worlds, kind: str, *,
                ledger: ResourceLedger | None = None) -> dict:
    """Exact accuracy by world for one observation kind; targets stay scorer-side."""
    encoded = encode_rows(worlds, kind)
    with torch.no_grad():
        output = model(encoded['token_ids'], encoded['task_features'])
        predictions = output.logits.argmax(dim=-1)
    if ledger is not None:
        ledger.record_inference(encoded['token_ids'].shape[0],
                                encoded['sequence_positions'])
    correct = (predictions == encoded['targets'])
    per_world = {}
    for world_id in sorted(set(encoded['world_ids'])):
        index = torch.tensor([i for i, value in enumerate(encoded['world_ids'])
                              if value == world_id])
        per_world[world_id] = float(correct[index].float().mean())
    return {
        'kind': kind,
        'accuracy': float(correct.float().mean()),
        'per_world': per_world,
        'examples': int(correct.numel()),
        'predictions': predictions.tolist(),
        'targets': encoded['targets'].tolist(),
        'world_ids': list(encoded['world_ids']),
        'inputs': list(encoded['inputs']),
        'observations_sha256': example_digest(encoded),
    }


def demonstration_recall_control(worlds) -> dict:
    """Perfect demo lookup with a constant unseen fallback; zero parameters."""
    demo = {('demo', world.world_id, x): world.target(x)
            for world in worlds for x in world.demo_inputs}
    results = {}
    for kind in OBSERVATION_KINDS:
        encoded = encode_rows(worlds, kind)
        predictions = []
        for world_id, x in zip(encoded['world_ids'], encoded['inputs']):
            predictions.append(demo.get(('demo', world_id, x),
                                        TOKEN_BASE))
        targets = encoded['targets']
        results[kind] = float((torch.tensor(predictions) == targets)
                              .float().mean())
    return {
        'control': 'demonstration_recall_only',
        'parameters': 0,
        'demo_accuracy': results['demo'],
        'transfer_accuracy': results['transfer'],
        'composition_accuracy': results['composition'],
        'fallback': 'fixed_token',
        'exact_training_answer_in_transfer': False,
    }


def _paired_interval(deltas, *, draws: int = 2000, confidence: float = 0.95,
                     seed: int = 13) -> dict:
    _require(deltas, 'nonempty paired observations required')
    _require(all(math.isfinite(value) and -1 <= value <= 1 for value in deltas),
             'paired deltas must be finite in [-1, 1]')
    _require(draws >= 100 and 0 < confidence < 1, 'invalid interval parameters')
    values = list(deltas)
    mean = sum(values) / len(values)
    if max(values) == min(values):
        unseen = 1 - ((1 - confidence) / 2) ** (1 / len(values))
        return {'mean': mean, 'low': values[0] - (values[0] + 1) * unseen,
                'high': values[0] + (1 - values[0]) * unseen,
                'n': len(values), 'method': 'constant_sample_unseen_guard'}
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values)
                     for _ in range(draws))
    tail = (1 - confidence) / 2
    return {'mean': mean, 'low': samples[int(tail * draws)],
            'high': samples[min(draws - 1, int((1 - tail) * draws))],
            'n': len(values), 'method': 'paired_percentile_bootstrap'}


def evaluate_rule_learning(pre: dict, post: dict, no_update: dict,
                           fake_update: dict, *, gate: dict,
                           recall_control: dict | None = None,
                           post_demo: dict | None = None) -> dict:
    """Gate on unseen rule applications, not demonstration recall."""
    _require(pre['kind'] == post['kind'] == no_update['kind']
             == fake_update['kind'] == 'transfer',
             'rule-learning gate requires transfer scores')
    world_ids = sorted(set(pre['per_world']))
    _require(all(set(score['per_world']) == set(world_ids)
                 for score in (post, no_update, fake_update)),
             'paired worlds must match across arms')
    min_gain = float(gate['min_rule_transfer_gain'])
    alpha = float(gate['family_alpha'])
    confidence = 1 - alpha / 2  # two control comparisons, Holm-style split
    comparisons = {}
    for name, control in (('no_update', no_update), ('fake_update', fake_update)):
        deltas = [post['per_world'][wid] - control['per_world'][wid]
                  for wid in world_ids]
        comparisons[name] = _paired_interval(
            deltas, draws=int(gate['bootstrap_draws']), confidence=confidence)
    transfer_before = pre['accuracy']
    transfer_after = post['accuracy']
    point_gain = transfer_after - transfer_before
    passed = (point_gain >= min_gain
              and all(value['mean'] >= min_gain
                      and value['low'] > 0
                      for value in comparisons.values()))
    if passed:
        decision, reason = 'accepted', 'heldout_transfer_gain'
    elif (post_demo is not None
          and post_demo['accuracy'] >= 0.99
          and transfer_after <= 0.1):
        decision, reason = 'rejected', 'demonstration_recall_without_rule_transfer'
    else:
        decision, reason = 'inconclusive', 'heldout_transfer_gain_unresolved'
    return {
        'decision': decision,
        'reason': reason,
        'transfer_before': transfer_before,
        'transfer_after': transfer_after,
        'point_gain': point_gain,
        'comparisons': comparisons,
        'confidence': confidence,
        'family_alpha': alpha,
        'adjustment': 'holm_split_two_control_comparisons',
        'min_rule_transfer_gain': min_gain,
        'post_demonstration_recall': (post_demo['accuracy']
                                      if post_demo is not None else None),
        'demonstration_recall_alone_sufficient': False,
        'recall_control': recall_control,
        'transfer_targets_used_in_training': False,
        'claim': 'fixture_gate_only_not_verified_learning',
    }


# ---------------------------------------------------------------------------
# Learning-policy arms: identical observations, update counts and limits


@dataclass(frozen=True)
class LearningPolicy:
    policy_id: str
    role: str
    learning_rate: float
    source: str

    def to_dict(self) -> dict:
        return {'policy_id': self.policy_id, 'role': self.role,
                'learning_rate': self.learning_rate, 'source': self.source}


def _policy_arm_design(limits: ResourceLimits) -> dict:
    return {
        'observations': 'identical_example_digest_per_world',
        'update_counts': 'identical_optimizer_steps_per_world',
        'resource_limits': limits.to_dict(),
        'initial_parameters': 'identical_seed_per_world',
        'policy_scope': 'one_learning_rate_action',
        'eval_labels_used_for_policy': False,
    }


def _meta_adapt_score(world: RuleWorld, *, spec, seed: int,
                      learning_rate: float, update_steps: int,
                      limits: ResourceLimits, max_grad_norm: float,
                      audit_hidden: int, max_positions: int) -> dict:
    model = build_model(spec, core_impl='recurrent', steps=2, seed=seed,
                        audit_hidden=audit_hidden,
                        max_positions=max_positions)
    demos = encode_rows([world], 'demo')
    ledger, training = adapt_adapter(
        model, demos, learning_rate=learning_rate,
        update_steps=update_steps, limits=limits, seed=seed,
        max_grad_norm=max_grad_norm)
    score = score_model(model, [world], 'transfer')
    return {'world_id': world.world_id, 'accuracy': score['accuracy'],
            'training': training, 'accounting': ledger.as_dict(),
            'observations_sha256': training['observations_sha256']}


def select_learning_policy(meta_worlds, *, spec: CoreSpec,
                           learning_rate_grid, update_steps: int,
                           limits: ResourceLimits, frozen_learning_rate: float,
                           seed: int = 0, max_grad_norm: float = 1.0,
                           audit_hidden: int = 16,
                           max_positions: int = DEFAULT_SEQUENCE) -> dict:
    """Bounded one-action policy selection on meta-training rule worlds."""
    _require(meta_worlds, 'meta-training worlds required')
    _require(learning_rate_grid and all(
        isinstance(value, (int, float)) and value > 0
        for value in learning_rate_grid), 'positive policy grid required')
    _require(len({float(value) for value in learning_rate_grid})
             == len(learning_rate_grid), 'policy grid must be distinct')
    _require(type(update_steps) is int and update_steps >= 1,
             'policy selection requires at least one update')
    candidates = []
    for rate in learning_rate_grid:
        results = [_meta_adapt_score(
            world, spec=spec, seed=seed + index, learning_rate=float(rate),
            update_steps=update_steps, limits=limits,
            max_grad_norm=max_grad_norm, audit_hidden=audit_hidden,
            max_positions=max_positions)
            for index, world in enumerate(meta_worlds)]
        mean = sum(result['accuracy'] for result in results) / len(results)
        candidates.append({'learning_rate': float(rate), 'mean_transfer': mean,
                           'worlds': results})
    # Deterministic tie-break toward the smaller rate.
    best = max(candidates, key=lambda item: (item['mean_transfer'],
                                             -item['learning_rate']))
    frozen = LearningPolicy('policy.frozen.v1', 'frozen',
                            float(frozen_learning_rate),
                            'declared default; no meta-selection')
    updated = LearningPolicy('policy.updated.v1', 'updated',
                             best['learning_rate'],
                             'meta-training transfer selection')
    total_meta_steps = len(candidates) * len(meta_worlds) * update_steps
    return {
        'format': POLICY_FORMAT,
        'frozen': frozen.to_dict(),
        'updated': updated.to_dict(),
        'candidates': candidates,
        'selected_learning_rate': best['learning_rate'],
        'meta_learning_rate_grid': [float(value) for value in learning_rate_grid],
        'meta_worlds': [world.world_id for world in meta_worlds],
        'meta_update_steps': update_steps,
        'meta_total_optimizer_steps': total_meta_steps,
        'meta_resource_limits': limits.to_dict(),
        'policy_update_uses_eval_labels': False,
        'claim': 'bounded_policy_selection_procedure_only',
    }


def compare_policy_arms(frozen: LearningPolicy, updated: LearningPolicy, worlds, *,
                        spec: CoreSpec, update_steps: int,
                        limits: ResourceLimits, seed: int = 0,
                        max_grad_norm: float = 1.0,
                        audit_hidden: int = 16,
                        max_positions: int = DEFAULT_SEQUENCE,
                        gate: dict | None = None) -> dict:
    """Paired frozen/updated arms with exact observation/update/limit equality."""
    _require(frozen.role == 'frozen' and updated.role == 'updated',
             'policy roles must be frozen and updated')
    _require(worlds, 'evaluation worlds required')
    arms = {}
    for policy in (frozen, updated):
        arm = {'policy': policy.to_dict(), 'worlds': {}, 'counters': [],
               'limits': [], 'observations': [], 'update_counts': [],
               'initial_digests': []}
        for index, world in enumerate(worlds):
            model = build_model(spec, core_impl='recurrent', steps=2,
                                seed=seed + index, audit_hidden=audit_hidden,
                                max_positions=max_positions)
            initial_digest = parameter_state_digest(model)
            pre_ledger = ResourceLedger(limits)
            pre = score_model(model, [world], 'transfer', ledger=pre_ledger)
            demos = encode_rows([world], 'demo')
            ledger, training = adapt_adapter(
                model, demos, learning_rate=policy.learning_rate,
                update_steps=update_steps, limits=limits, seed=seed + index,
                max_grad_norm=max_grad_norm)
            post_ledger = ResourceLedger(limits)
            post = score_model(model, [world], 'transfer', ledger=post_ledger)
            counters = {key: pre_ledger.counters[key] + ledger.counters[key]
                        + post_ledger.counters[key] for key in COUNTER_KEYS}
            arm['worlds'][world.world_id] = {
                'pre_accuracy': pre['accuracy'],
                'post_accuracy': post['accuracy'],
                'gain': post['accuracy'] - pre['accuracy'],
                'observations_sha256': training['observations_sha256'],
                'initial_parameters_sha256': initial_digest,
                'update_steps': update_steps,
                'counters': counters,
                'limits': limits.to_dict(),
            }
            arm['observations'].append(training['observations_sha256'])
            arm['update_counts'].append(update_steps)
            arm['initial_digests'].append(initial_digest)
            arm['limits'].append(limits.to_dict())
        arms[policy.role] = arm

    frozen_arm, updated_arm = arms['frozen'], arms['updated']
    matched = {
        'observations': frozen_arm['observations'] == updated_arm['observations'],
        'update_counts': frozen_arm['update_counts'] == updated_arm['update_counts'],
        'resource_limits': frozen_arm['limits'] == updated_arm['limits'],
        'initial_parameters': frozen_arm['initial_digests'] == updated_arm['initial_digests'],
    }
    world_ids = sorted(frozen_arm['worlds'])
    deltas = [updated_arm['worlds'][wid]['gain']
              - frozen_arm['worlds'][wid]['gain'] for wid in world_ids]
    gate = gate or {'min_rule_transfer_gain': 0.05,
                    'family_alpha': 0.05, 'bootstrap_draws': 2000}
    interval = _paired_interval(deltas, draws=int(gate['bootstrap_draws']),
                                confidence=1 - float(gate['family_alpha']) / 2)
    all_matched = all(matched.values())
    passed = all_matched and interval['mean'] >= float(
        gate['min_rule_transfer_gain']) and interval['low'] > 0
    return {
        'format': POLICY_FORMAT,
        'arms': arms,
        'matched': matched,
        'all_matched': all_matched,
        'design': _policy_arm_design(limits),
        'paired_gain_delta': interval,
        'decision': 'accepted' if passed else 'inconclusive',
        'reason': ('matched_policy_gain' if passed
                   else 'insufficient_or_unmatched_policy_gain'),
        'update_steps': update_steps,
        'claim': 'bounded_policy_procedure_only_not_rsi',
        'policy_update_cost_excluded_from_paired_arms': True,
    }


# ---------------------------------------------------------------------------
# Detached audit and separate personality fidelity


def audit_objective(model: RuleResponseModel, worlds, *, seed: int = 0) -> Tensor:
    """One detached monitor objective; adapter gradients must stay absent."""
    encoded = encode_rows(worlds, 'demo')
    output = model.audit_logits(encoded['token_ids'], encoded['task_features'])
    targets = (encoded['targets'] == encoded['targets'][0]).float()
    return audit_cross_entropy(output, targets)


def style_trace(values, *, style_values: int = 4) -> tuple[float, ...]:
    """Normalized synthetic style-marker frequencies; no factual content."""
    values = list(values)
    _require(values and style_values >= 1, 'nonempty style values required')
    _require(all(type(value) is int and 0 <= value < style_values
                 for value in values), 'style values outside declared range')
    counts = [0] * style_values
    for value in values:
        counts[value] += 1
    return tuple(count / len(values) for count in counts)


def personality_fidelity(predicted_traces, reference_traces) -> dict:
    """Cosine similarity of synthetic style traces, kept outside accuracy."""
    predicted = [torch.tensor(trace, dtype=torch.float32)
                 for trace in predicted_traces]
    reference = [torch.tensor(trace, dtype=torch.float32)
                 for trace in reference_traces]
    _require(predicted and len(predicted) == len(reference),
             'paired style traces required')
    _require(all(a.shape == b.shape for a, b in zip(predicted, reference)),
             'style trace shapes must match')
    scores = []
    for left, right in zip(predicted, reference):
        _finite(left, 'predicted style trace')
        _finite(right, 'reference style trace')
        _require(bool((left >= 0).all() and (right >= 0).all()),
                 'style traces must be nonnegative')
        denominator = left.norm() * right.norm()
        scores.append(float((left @ right) / denominator) if denominator > 0
                      else 1.0)
    return {
        'metric': 'cosine_synthetic_style_trace',
        'mean': sum(scores) / len(scores),
        'per_trace': scores,
        'factual_accuracy_used': False,
        'memory_accuracy_used': False,
        'authentic_personality_claim': False,
        'status': 'synthetic_style_only',
    }


# ---------------------------------------------------------------------------
# Protocol, deterministic preparation and the CPU fixture report


def _implementation_pins() -> dict:
    pins = {}
    for relative in IMPLEMENTATION_PATHS:
        path = PROJECT_ROOT / relative
        if path.is_file():
            pins[relative] = _sha_file(path)
    return pins


def build_protocol(worlds, config: dict) -> dict:
    _require(config.get('format') == CONFIG_FORMAT,
             f'configuration format must be {CONFIG_FORMAT}')
    spec = CoreSpec(width=config['model']['width'],
                    slots=config['model']['slots'],
                    steps=2, heads=config['model']['heads'],
                    task_features=config['model']['task_features'],
                    ffn_hidden=config['model']['ffn_hidden'],
                    control_hidden=config['model']['control_hidden'])
    controls = build_control_plan(spec)
    limits = ResourceLimits.from_dict(config['limits'])
    protocol = {
        'format': PROTOCOL_FORMAT,
        'mode': 'cpu_fixture_contract',
        'native_evaluation': dict(config['native_evaluation']),
        'claims': dict(config['claims']),
        'objective': {
            'loss': 'response_cross_entropy_through_frozen_base_decoder',
            'training_rows': 'demonstrations_only',
            'transfer_targets_used_in_training': False,
            'audit_logits_rejected_by_objective': True,
            'monitor_detached': True,
        },
        'worlds': {
            'seed': config['seed'],
            'families_per_split': dict(config['worlds']['families_per_split']),
            'modulus': config['worlds']['modulus'],
            'demo_count': config['worlds']['demo_count'],
            'counts': {split: sum(1 for world in worlds if world.split == split)
                       for split in SPLITS},
            'observation_sha256': observation_digest(worlds),
            'families': [world.family for world in worlds],
        },
        'model': dict(config['model']),
        'training': dict(config['training']),
        'limits': limits.to_dict(),
        'controls': controls,
        'policy': dict(config['policy']),
        'gates': dict(config['gates']),
        'fidelity': dict(config['fidelity']),
        'implementation_sha256': _implementation_pins(),
        'config_sha256': _sha_json(config),
    }
    protocol['protocol_sha256'] = _sha_json(protocol)
    return protocol


def default_config() -> dict:
    return {
        'format': CONFIG_FORMAT,
        'declared_at_utc': '2026-10-25T00:00:00+00:00',
        'purpose': 'CPU procedure contract binding held-out rule transfer, matched learning-policy arms, controls, detached audit and exact accounting.',
        'seed': 20261025,
        'worlds': {
            'modulus': 10,
            'families_per_split': {'train': 4, 'dev': 3, 'heldout': 4},
            'demo_count': 4,
            'composition_count': 2,
            'style_width': 4,
            'style_values': 4,
        },
        'model': {
            'width': 16, 'slots': 2, 'heads': 2, 'task_features': 4,
            'ffn_hidden': 32, 'control_hidden': 32, 'audit_hidden': 16,
            'max_positions': DEFAULT_SEQUENCE,
        },
        'training': {
            'learning_rate': 0.05, 'update_steps': 4,
            'max_grad_norm': 1.0, 'fake_target_shift': 1,
        },
        'limits': ResourceLimits(
            max_forward_batches=256, max_forward_examples=8192,
            max_forward_positions=65536, max_backward_batches=256,
            max_inference_batches=512, max_inference_examples=16384,
            max_inference_positions=131072, max_optimizer_steps=128,
            max_gradient_clip_calls=256, max_adapter_parameters=25000,
            max_optimizer_state_elements=0,
            max_parameter_delta_norm=100.0).to_dict(),
        'policy': {
            'learning_rate_grid': [0.01, 0.05, 0.1],
            'frozen_learning_rate': 0.05,
            'update_steps': 2,
            'meta_split': 'train',
            'comparison_split': 'heldout',
        },
        'gates': {
            'min_rule_transfer_gain': 0.05,
            'family_alpha': 0.05,
            'bootstrap_draws': 2000,
            'min_personality_fidelity': 0.9,
            'retention_gate': 'required_separately',
        },
        'fidelity': {
            'style_values': 4,
            'claim': 'synthetic_style_only_personality_specification_pending',
        },
        'claims': {'verified_learning': False, 'rsi': False,
                   'native_capability': False},
        'native_evaluation': {
            'required': True, 'status': 'not_run',
            'gate': 'separately_gated_native_experiment',
            'reason': 'CPU fixtures cannot establish verified learning or RSI',
        },
    }


def load_config(path) -> dict:
    value = json.loads(Path(path).read_text())
    _require(value.get('format') == CONFIG_FORMAT,
             f'configuration format must be {CONFIG_FORMAT}')
    _require(value.get('claims', {}).get('verified_learning') is False
             and value['claims'].get('rsi') is False,
             'configuration may not pre-claim verified learning or RSI')
    ResourceLimits.from_dict(value['limits'])
    return value


def _worlds_from_config(config: dict) -> list[RuleWorld]:
    world_config = config['worlds']
    return generate_rule_worlds(
        seed=config['seed'], modulus=world_config['modulus'],
        families_per_split=world_config['families_per_split'],
        demo_count=world_config['demo_count'],
        composition_count=world_config.get('composition_count', 2),
        style_width=world_config.get('style_width', 4),
        style_values=world_config.get('style_values', 4))


def prepare_verified_learning(output, *, config_path=None, config=None) -> dict:
    """Write deterministic worlds/protocol pins without a model or native job."""
    config = dict(config or load_config(config_path))
    _require(config.get('format') == CONFIG_FORMAT,
             f'configuration format must be {CONFIG_FORMAT}')
    output = Path(output)
    _require(not output.exists() or not any(output.iterdir()),
             'preparation output must be fresh')
    output.mkdir(parents=True, exist_ok=True)
    worlds = _worlds_from_config(config)
    protocol = build_protocol(worlds, config)
    worlds_payload = {
        'format': PREPARED_FORMAT,
        'seed': config['seed'],
        'worlds': [world.to_dict() for world in worlds],
    }
    protocol_path = output / 'protocol.json'
    worlds_path = output / 'worlds.json'
    if config_path is not None:
        declared = json.loads(Path(config_path).read_text())
        _require(declared == config, 'in-memory config differs from declared config')
    _write_json(worlds_path, worlds_payload)
    _write_json(protocol_path, protocol)
    manifest = {
        'format': PREPARED_FORMAT,
        'protocol_sha256': protocol['protocol_sha256'],
        'worlds_sha256': _sha_file(worlds_path),
        'config_sha256': _sha_file(config_path) if config_path else _sha_json(config),
        'observation_sha256': protocol['worlds']['observation_sha256'],
        'implementation_sha256': protocol['implementation_sha256'],
        'worlds': len(worlds),
        'counts': protocol['worlds']['counts'],
        'purpose': 'deterministic CPU procedure fixture; native evaluation not run',
    }
    _write_json(output / 'manifest.json', manifest)
    return manifest


def verify_prepared(prepared) -> dict:
    prepared = Path(prepared)
    manifest_path = prepared / 'manifest.json'
    protocol_path = prepared / 'protocol.json'
    worlds_path = prepared / 'worlds.json'
    _require(manifest_path.is_file() and protocol_path.is_file()
             and worlds_path.is_file(), 'prepared inputs are incomplete')
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    worlds_payload = json.loads(worlds_path.read_text())
    _require(manifest.get('format') == PREPARED_FORMAT
             and protocol.get('format') == PROTOCOL_FORMAT
             and worlds_payload.get('format') == PREPARED_FORMAT,
             'prepared formats differ')
    _require(protocol['protocol_sha256'] == _sha_json(
        {key: value for key, value in protocol.items()
         if key != 'protocol_sha256'}), 'protocol hash does not match bytes')
    _require(manifest['protocol_sha256'] == protocol['protocol_sha256'],
             'manifest protocol pin differs')
    _require(manifest['worlds_sha256'] == _sha_file(worlds_path),
             'world bytes differ from manifest')
    worlds = [RuleWorld.from_dict(value) for value in worlds_payload['worlds']]
    _require(observation_digest(worlds) == manifest['observation_sha256'],
             'observation digest differs from manifest')
    pins = manifest.get('implementation_sha256')
    _require(isinstance(pins, dict) and pins, 'implementation pins are missing')
    _require(pins == _implementation_pins(),
             'implementation pins differ from current source bytes')
    return {'manifest': manifest, 'protocol': protocol, 'worlds': worlds,
            'protocol_sha256': protocol['protocol_sha256']}


def run_fixture(config: dict, *, device: str = 'cpu') -> dict:
    """Exercise the CPU procedure only; native execution is separately gated."""
    _require(device == 'cpu',
             'native evaluation is separately gated; this runner is CPU-only')
    worlds = _worlds_from_config(config)
    train = [world for world in worlds if world.split == 'train']
    heldout = [world for world in worlds if world.split == 'heldout']
    protocol = build_protocol(worlds, config)
    spec = CoreSpec(width=config['model']['width'],
                    slots=config['model']['slots'], steps=2,
                    heads=config['model']['heads'],
                    task_features=config['model']['task_features'],
                    ffn_hidden=config['model']['ffn_hidden'],
                    control_hidden=config['model']['control_hidden'])
    limits = ResourceLimits.from_dict(config['limits'])
    training = config['training']
    model = build_model(spec, core_impl='recurrent', steps=2,
                        seed=config['seed'], audit_hidden=config['model']['audit_hidden'],
                        max_positions=config['model']['max_positions'])
    pre = score_model(model, heldout, 'transfer')
    demos = encode_rows(train, 'demo')
    ledger, training_summary = adapt_adapter(
        model, demos, learning_rate=training['learning_rate'],
        update_steps=training['update_steps'], limits=limits,
        seed=config['seed'], max_grad_norm=training['max_grad_norm'])
    post = score_model(model, heldout, 'transfer')
    demo_post = score_model(model, train, 'demo')
    composition_post = score_model(model, heldout, 'composition')

    fake_model = build_model(spec, core_impl='recurrent', steps=2,
                             seed=config['seed'],
                             audit_hidden=config['model']['audit_hidden'],
                             max_positions=config['model']['max_positions'])
    fake_ledger, fake_summary = adapt_adapter(
        fake_model, demos, learning_rate=training['learning_rate'],
        update_steps=training['update_steps'], limits=limits,
        seed=config['seed'], max_grad_norm=training['max_grad_norm'],
        fake_targets=True)
    fake = score_model(fake_model, heldout, 'transfer')
    no_update = pre
    recall = demonstration_recall_control(heldout)
    rule_gate = evaluate_rule_learning(
        pre, post, no_update, fake, gate=config['gates'],
        recall_control=recall, post_demo=demo_post)

    policy_config = config['policy']
    _require(policy_config['meta_split'] in SPLITS
             and policy_config['comparison_split'] in SPLITS,
             'policy splits must be declared world splits')
    meta_worlds = [world for world in worlds
                   if world.split == policy_config['meta_split']]
    comparison_worlds = [world for world in worlds
                         if world.split == policy_config['comparison_split']]
    _require(meta_worlds and comparison_worlds,
             'policy meta/comparison splits must be nonempty')
    policy_update = select_learning_policy(
        meta_worlds,
        spec=spec, learning_rate_grid=policy_config['learning_rate_grid'],
        update_steps=policy_config['update_steps'], limits=limits,
        frozen_learning_rate=policy_config['frozen_learning_rate'],
        seed=config['seed'], max_grad_norm=training['max_grad_norm'],
        audit_hidden=config['model']['audit_hidden'],
        max_positions=config['model']['max_positions'])
    frozen = LearningPolicy(**policy_update['frozen'])
    updated = LearningPolicy(**policy_update['updated'])
    policy_comparison = compare_policy_arms(
        frozen, updated, comparison_worlds, spec=spec,
        update_steps=policy_config['update_steps'], limits=limits,
        seed=config['seed'], max_grad_norm=training['max_grad_norm'],
        audit_hidden=config['model']['audit_hidden'],
        max_positions=config['model']['max_positions'],
        gate=config['gates'])

    style_values = config['fidelity']['style_values']
    reference = [style_trace(world.style, style_values=style_values)
                 for world in heldout]
    predicted = [style_trace(world.style, style_values=style_values)
                 for world in heldout]
    fidelity = personality_fidelity(predicted, reference)

    audit_encoded = encode_rows(heldout, 'demo')
    audit_output = model.audit_logits(audit_encoded['token_ids'],
                                      audit_encoded['task_features'])
    try:
        response_cross_entropy(audit_output, audit_encoded['targets'])
        audit_rejected = False
    except TypeError:
        audit_rejected = True
    _require(audit_rejected, 'audit logits must be rejected by the objective')

    report = {
        'format': FIXTURE_FORMAT,
        'mode': 'cpu_procedure_fixture',
        'device': 'cpu',
        'protocol_sha256': protocol['protocol_sha256'],
        'claims': dict(config['claims']),
        'native_evaluation': dict(config['native_evaluation']),
        'worlds': protocol['worlds'],
        'training': training_summary,
        'fake_training': {**fake_summary,
                          'ledger_matches_real_counter_shape':
                          (fake_ledger.counters == ledger.counters)},
        'rule_learning': {
            'pre_transfer': pre,
            'post_transfer': post,
            'post_demonstration_recall': demo_post,
            'post_composition': composition_post,
            'fake_transfer': fake,
            'gate': rule_gate,
        },
        'policy': {
            'selection': policy_update,
            'comparison': policy_comparison,
        },
        'controls': protocol['controls'],
        'audit': {
            'head': 'TaskSuccessHead',
            'detached_from_adapter': True,
            'adapter_trainable': training_summary['audit_trainable'],
            'parameters': parameter_count(model.audit),
            'logits_finite': bool(torch.isfinite(audit_output.logits).all()),
            'response_objective_rejected': audit_rejected,
            'carries_hidden_answers': False,
            'substitutes_for_objective': False,
        },
        'fidelity': {
            **fidelity,
            'factual_accuracy': {kind: score_model(model, heldout, kind)['accuracy']
                                 for kind in OBSERVATION_KINDS},
            'memory_accuracy': None,
            'memory_accuracy_status': 'not_measured_no_memory_bank_in_fixture',
            'combined_score': None,
            'min_required': config['gates']['min_personality_fidelity'],
            'authentic_personality_content': 'pending_user_specification',
        },
        'accounting': {
            'real': ledger.as_dict(),
            'fake': fake_ledger.as_dict(),
            'exact_counters': True,
            'wall_time_reported': False,
        },
        'status': 'procedure_only',
        'remaining_prerequisites': [
            'independently reviewed CPU component tests',
            'native capability experiment under separate gates',
            'retention/replay measurement against general regression tasks',
            'permitted personality specification and human fidelity review',
        ],
    }
    return report
