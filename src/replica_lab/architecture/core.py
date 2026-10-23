"""E3 shared cognitive core: a tied recurrent workspace block and matched control.

The core owns no request state: every tensor enters through the call and every
tensor leaves as a value. There is no storage in forward, no module-level
workspace, and no detach between tied iterations. A fully masked source is
dropped from the key/value set entirely, so its contribution is exactly zero.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import CoreSpec

SOURCES = ('sense', 'memory', 'task')


def _finite(value: Tensor, name: str) -> None:
    if not value.is_floating_point():
        raise ValueError(f'{name} must be floating point')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')


def core_parameter_count(module: nn.Module) -> int:
    """Total parameter count of one core/control module; test-facing helper."""
    return sum(parameter.numel() for parameter in module.parameters())


def _validate_source(value: Tensor, width: int, name: str) -> None:
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != width:
        raise ValueError(f'{name} must have shape [1, positions, {width}]')
    _finite(value, name)


def _validate_valid(value: Tensor, source: Tensor, name: str) -> Tensor:
    if value.dtype != torch.bool or tuple(value.shape) != tuple(source.shape[:2]):
        raise ValueError(f'{name} must be a boolean mask matching [1, positions]')
    return value.to(device=source.device)


class TaskStateEncoder(nn.Module):
    """Host-computed exact task features to two workspace-width tokens.

    The input is the bounded [0, 1] feature vector from the exact journal; this
    module adds a learned projection and norm only. Two tokens are formed from
    one projection so the same linear map encodes both task tokens.
    """

    def __init__(self, features: int, width: int) -> None:
        super().__init__()
        if features < 1 or width < 1:
            raise ValueError('Positive feature and width dimensions required')
        self.features = features
        self.width = width
        self.projection = nn.Linear(features, 2 * width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, task_features: Tensor) -> Tensor:
        if task_features.ndim != 2 or task_features.shape[0] != 1 \
                or task_features.shape[1] != self.features:
            raise ValueError(f'task_features must have shape [1, {self.features}]')
        _finite(task_features, 'task_features')
        tokens = self.projection(task_features).reshape(1, 2, self.width)
        return self.norm(tokens)


class CognitiveCore(nn.Module):
    """Tied shared core: ``steps`` iterations of one parameter block.

    Each iteration applies, with identical parameters: slot self-attention under
    the validity mask, masked cross-attention over sense summaries, memory slots
    and the two encoded task tokens (distinguished by learned type embeddings),
    and a pre-norm gated residual feed-forward with a sigmoid gate per sublayer.
    The initial workspace is the learned slot queries plus the task-owned
    previous workspace (zero when absent). No hidden-state detach occurs between
    iterations; ``steps=0`` returns the initial workspace unchanged.
    """

    def __init__(self, spec: CoreSpec) -> None:
        super().__init__()
        for name in ('width', 'slots', 'heads', 'task_features', 'ffn_hidden', 'control_hidden'):
            value = getattr(spec, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'CoreSpec.{name} must be a positive integer')
        if spec.width % spec.heads:
            raise ValueError('Core width must divide heads')
        if type(spec.steps) is not int or not 0 <= spec.steps <= 8:
            raise ValueError('CoreSpec.steps must be between 0 and 8')
        self.spec = spec
        width = spec.width
        self.width, self.slots, self.heads, self.head_size = width, spec.slots, spec.heads, width // spec.heads
        self.slot_queries = nn.Parameter(torch.randn(spec.slots, width) * 0.02)
        self.type_embeddings = nn.Parameter(torch.randn(3, width) * 0.02)
        self.self_norm = nn.LayerNorm(width)
        self.cross_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.self_q = nn.Linear(width, width, bias=False)
        self.self_k = nn.Linear(width, width, bias=False)
        self.self_v = nn.Linear(width, width, bias=False)
        self.self_o = nn.Linear(width, width, bias=False)
        self.cross_q = nn.Linear(width, width, bias=False)
        self.cross_k = nn.Linear(width, width, bias=False)
        self.cross_v = nn.Linear(width, width, bias=False)
        self.cross_o = nn.Linear(width, width, bias=False)
        # The recurrent feed-forward is ffn_hidden (1024 at the default spec)
        # wide; the matched control below matches the *total* core parameters.
        self.ff_up = nn.Linear(width, spec.ffn_hidden, bias=False)
        self.ff_down = nn.Linear(spec.ffn_hidden, width, bias=False)
        self.task_encoder = TaskStateEncoder(spec.task_features, width)
        self.gate_self_logit = nn.Parameter(torch.zeros(()))
        self.gate_cross_logit = nn.Parameter(torch.zeros(()))
        self.gate_ffn_logit = nn.Parameter(torch.zeros(()))

    def _attend(self, query_source: Tensor, key_source: Tensor,
                valid: Tensor | None) -> Tensor:
        batch, length, width = query_source.shape
        count = key_source.shape[1]
        query = self.self_q(query_source) if key_source is query_source else self.cross_q(query_source)
        key = self.self_k(key_source) if key_source is query_source else self.cross_k(key_source)
        value = self.self_v(key_source) if key_source is query_source else self.cross_v(key_source)
        query = query.reshape(batch, length, self.heads, self.head_size).transpose(1, 2)
        key = key.reshape(batch, count, self.heads, self.head_size).transpose(1, 2)
        value = value.reshape(batch, count, self.heads, self.head_size).transpose(1, 2)
        mask = None
        available = None
        if valid is not None:
            # Keep softmax well-defined for a fully invalid row, then remove the
            # entire read below so an all-masked source contributes exactly zero
            # rather than attending to an arbitrary first key.
            safe = valid.clone()
            available = valid.any(dim=1, keepdim=True)
            safe[:, 0] = safe[:, 0] | ~available.squeeze(1)
            mask = safe[:, None, None, :]
        read = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
        if available is not None:
            read = read.masked_fill(~available[:, None, None], 0)
        output = read.transpose(1, 2).reshape(batch, length, width)
        return self.self_o(output) if key_source is query_source else self.cross_o(output)

    def _sublayer(self, workspace: Tensor, update: Tensor, logit: Tensor) -> Tensor:
        return workspace + torch.sigmoid(logit) * update

    def forward(self, sense: Tensor, sense_valid: Tensor, memory: Tensor,
                memory_valid: Tensor, task_features: Tensor, *, previous: Tensor | None = None,
                enabled_inputs: tuple[str, ...] = SOURCES, steps: int | None = None) -> tuple[Tensor, Tensor]:
        if tuple(enabled_inputs) and (set(enabled_inputs) - set(SOURCES)
                                      or len(set(enabled_inputs)) != len(enabled_inputs)):
            raise ValueError('enabled_inputs must be a distinct subset of sense/memory/task')
        _validate_source(sense, self.width, 'sense')
        _validate_source(memory, self.width, 'memory')
        sense_valid = _validate_valid(sense_valid, sense, 'sense_valid')
        memory_valid = _validate_valid(memory_valid, memory, 'memory_valid')
        if previous is not None:
            _validate_source(previous, self.width, 'previous')
            if previous.shape[1] != self.slots:
                raise ValueError(f'previous must have shape [1, {self.slots}, {self.width}]')
        steps = self.spec.steps if steps is None else steps
        if type(steps) is not int or not 0 <= steps <= 8:
            raise ValueError('steps must be between 0 and 8')

        workspace = self.slot_queries.unsqueeze(0)
        if previous is not None:
            workspace = workspace + previous
        valid = torch.ones(1, self.slots, dtype=torch.bool, device=workspace.device)
        if steps == 0:
            _finite(workspace, 'initial workspace')
            return workspace, valid

        sources, source_valid = [], []
        if 'sense' in enabled_inputs and sense.shape[1] and bool(sense_valid.any()):
            sources.append(sense + self.type_embeddings[0])
            source_valid.append(sense_valid)
        if 'memory' in enabled_inputs and memory.shape[1] and bool(memory_valid.any()):
            sources.append(memory + self.type_embeddings[1])
            source_valid.append(memory_valid)
        if 'task' in enabled_inputs:
            sources.append(self.task_encoder(task_features) + self.type_embeddings[2])
            source_valid.append(torch.ones(1, 2, dtype=torch.bool, device=workspace.device))
        if sources:
            keys = self.source_norm(torch.cat(sources, dim=1).masked_fill(
                ~torch.cat(source_valid, dim=1).unsqueeze(-1), 0))
            keys_valid = torch.cat(source_valid, dim=1)

        for _ in range(steps):
            normalized = self.self_norm(workspace)
            workspace = self._sublayer(workspace, self._attend(normalized, normalized, valid),
                                       self.gate_self_logit)
            if sources:
                workspace = self._sublayer(workspace, self._attend(
                    self.cross_norm(workspace), keys, keys_valid), self.gate_cross_logit)
            workspace = self._sublayer(workspace, self.ff_down(
                F.silu(self.ff_up(self.ffn_norm(workspace)))), self.gate_ffn_logit)
        _finite(workspace, 'core workspace')
        return workspace, valid


def cognitive_parameter_count(spec: CoreSpec) -> int:
    """Pure analytic parameter count of ``CognitiveCore`` for the given spec."""
    width = spec.width
    return (spec.slots * width          # learned slot queries
            + 3 * width                 # type embeddings
            + 4 * 2 * width             # self/cross/ffn/source norms
            + 8 * width * width         # self and cross q/k/v/o
            + 2 * width * spec.ffn_hidden
            + 2 * width * spec.task_features + 2 * width  # task encoder
            + 3)                        # one sigmoid gate per sublayer


class FeedForwardCore(nn.Module):
    """Matched control: one gated feed-forward from pooled features to slots.

    Maps pooled sense and memory features plus the exact task features to the
    same workspace contract as ``CognitiveCore``. The hidden width is
    ``control_hidden``; with the default spec (340) the control's *total*
    parameter count stays within 5% of the recurrent core's total (not just its
    feed-forward arm), which is asserted at construction.
    """

    def __init__(self, spec: CoreSpec) -> None:
        super().__init__()
        self.spec = spec
        self.width, self.slots = spec.width, spec.slots
        self.in_dim = 2 * spec.width + spec.task_features
        self.out_dim = spec.slots * spec.width
        self.ff_up = nn.Linear(self.in_dim, spec.control_hidden, bias=False)
        self.ff_gate = nn.Linear(self.in_dim, spec.control_hidden, bias=False)
        self.ff_down = nn.Linear(spec.control_hidden, self.out_dim, bias=False)
        target = cognitive_parameter_count(spec)
        actual = core_parameter_count(self)
        if not 0.95 <= actual / target <= 1.05:
            raise ValueError('Matched-control parameter budget exceeded 5% of the recurrent core')

    def forward(self, pooled_sense: Tensor, pooled_memory: Tensor,
                task_features: Tensor) -> tuple[Tensor, Tensor]:
        for value, name, expected in ((pooled_sense, 'pooled_sense', self.width),
                                      (pooled_memory, 'pooled_memory', self.width),
                                      (task_features, 'task_features', self.spec.task_features)):
            if value.ndim != 2 or value.shape[0] != 1 or value.shape[-1] != expected:
                raise ValueError(f'{name} must have shape [1, {expected}]')
            _finite(value, name)
        if pooled_sense.device != pooled_memory.device or pooled_sense.device != task_features.device:
            raise ValueError('Control inputs must share one device')
        features = torch.cat((pooled_sense, pooled_memory, task_features), dim=-1)
        hidden = F.silu(self.ff_up(features)) * torch.sigmoid(self.ff_gate(features))
        workspace = self.ff_down(hidden).reshape(1, self.slots, self.width)
        _finite(workspace, 'control workspace')
        return workspace, torch.ones(1, self.slots, dtype=torch.bool, device=workspace.device)
