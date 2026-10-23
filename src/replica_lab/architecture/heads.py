"""E3 prediction monitor head. A monitor logit is never a task fact."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .contracts import CoreSpec


class TaskSuccessHead(nn.Module):
    """One logit from ``[task_features, pooled workspace]``.

    The input vector is detached before the MLP, so this monitor never feeds a
    gradient back into the core, task encoder or workspace. The returned logit
    stays attached to this head's own parameters: a separate training driver may
    backpropagate a supervised success objective into the head only. The head
    never authorizes a success claim, and no calibration property is claimed or
    implied.
    """

    def __init__(self, spec: CoreSpec, hidden: int = 64) -> None:
        super().__init__()
        for name in ('width', 'slots', 'task_features'):
            value = getattr(spec, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'CoreSpec.{name} must be a positive integer')
        if type(hidden) is not int or hidden < 1:
            raise ValueError('hidden must be a positive integer')
        self.spec = spec
        self.mlp = nn.Sequential(
            nn.Linear(spec.task_features + spec.width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, task_features: Tensor, workspace: Tensor,
                valid: Tensor | None = None) -> Tensor:
        if task_features.ndim != 2 or task_features.shape[0] != 1 \
                or task_features.shape[1] != self.spec.task_features:
            raise ValueError(f'task_features must have shape [1, {self.spec.task_features}]')
        if workspace.ndim != 3 or workspace.shape[0] != 1 \
                or workspace.shape[1] != self.spec.slots or workspace.shape[-1] != self.spec.width:
            raise ValueError(f'workspace must have shape [1, {self.spec.slots}, {self.spec.width}]')
        for value, name in ((task_features, 'task_features'), (workspace, 'workspace')):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite floating point')
        if valid is None:
            valid = torch.ones(1, self.spec.slots, dtype=torch.bool, device=workspace.device)
        if valid.dtype != torch.bool or tuple(valid.shape) != tuple(workspace.shape[:2]):
            raise ValueError('valid must be a boolean mask matching [batch, slots]')
        if task_features.device != workspace.device:
            raise ValueError('task_features and workspace must be on the same device')
        # Detach only the inputs: the monitor must not train the core, while the
        # head itself remains trainable by a separate supervised objective.
        features = task_features.detach()
        slots = workspace.detach()
        mask = valid.to(device=slots.device)
        pooled = slots.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / \
            mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.mlp(torch.cat((features, pooled), dim=-1)).squeeze(-1)


ACTION_KINDS = ('read', 'run_test', 'set_constant', 'stop')


class ActionKindHead(nn.Module):
    """Predict a bounded next-action kind from prompt-only core state.

    Inputs are detached so the head never trains the core or task encoder. The
    output is a proposal prior, never authority: the runtime still validates the
    selected kind, the model must generate its arguments, and the host may deny
    the action. The head does not write to memory or decide task success.
    """

    def __init__(self, spec: CoreSpec, hidden: int = 64,
                 kinds: tuple[str, ...] = ACTION_KINDS) -> None:
        super().__init__()
        if not kinds or len(set(kinds)) != len(kinds) or any(
                not isinstance(kind, str) or not kind for kind in kinds):
            raise ValueError('action kinds must be distinct nonempty strings')
        for name in ('width', 'slots', 'task_features'):
            value = getattr(spec, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'CoreSpec.{name} must be a positive integer')
        if type(hidden) is not int or hidden < 1:
            raise ValueError('hidden must be a positive integer')
        self.spec, self.kinds = spec, tuple(kinds)
        self.mlp = nn.Sequential(
            nn.Linear(spec.task_features + spec.width, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(self.kinds)),
        )

    def forward(self, task_features: Tensor, workspace: Tensor,
                valid: Tensor | None = None) -> Tensor:
        if task_features.ndim != 2 or task_features.shape[0] < 1 \
                or task_features.shape[1] != self.spec.task_features:
            raise ValueError(f'task_features must have shape [batch, {self.spec.task_features}]')
        if workspace.ndim != 3 or workspace.shape[0] != task_features.shape[0] \
                or workspace.shape[1] != self.spec.slots \
                or workspace.shape[-1] != self.spec.width:
            raise ValueError(f'workspace must have shape [batch, {self.spec.slots}, {self.spec.width}]')
        for value, name in ((task_features, 'task_features'), (workspace, 'workspace')):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite floating point')
        if valid is None:
            valid = torch.ones(workspace.shape[:2], dtype=torch.bool,
                               device=workspace.device)
        if valid.dtype != torch.bool or tuple(valid.shape) != tuple(workspace.shape[:2]):
            raise ValueError('valid must be a boolean mask matching [batch, slots]')
        if task_features.device != workspace.device:
            raise ValueError('task_features and workspace must be on the same device')
        mask = valid.to(workspace.device)
        pooled = workspace.detach().masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / \
            mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.mlp(torch.cat((task_features.detach(), pooled), dim=-1))


class PooledTaskOutcomeHead(nn.Module):
    """Outcome logit from exact task features plus the runtime's pooled workspace.

    This is the monitor shape emitted by ``TaskRuntime.summary().feature_vector``
    (24 task features + one pooled 256-wide workspace vector). It is trained only
    from actual episode outcomes, never declares success, and has no calibration
    authority. Inputs are detached so it cannot train the runtime or core.
    """

    def __init__(self, spec: CoreSpec, hidden: int = 64) -> None:
        super().__init__()
        for name in ('width', 'task_features'):
            value = getattr(spec, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'CoreSpec.{name} must be a positive integer')
        if type(hidden) is not int or hidden < 1:
            raise ValueError('hidden must be a positive integer')
        self.spec = spec
        self.mlp = nn.Sequential(
            nn.Linear(spec.task_features + spec.width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, task_features: Tensor, workspace_pool: Tensor) -> Tensor:
        if task_features.ndim != 2 or task_features.shape[1] != self.spec.task_features:
            raise ValueError(f'task_features must be [batch, {self.spec.task_features}]')
        if workspace_pool.ndim != 2 or workspace_pool.shape[0] != task_features.shape[0] \
                or workspace_pool.shape[1] != self.spec.width:
            raise ValueError(f'workspace_pool must be [batch, {self.spec.width}]')
        for value, name in ((task_features, 'task_features'), (workspace_pool, 'workspace_pool')):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must be finite floating point')
        if task_features.device != workspace_pool.device:
            raise ValueError('task_features and workspace_pool must share a device')
        return self.mlp(torch.cat((task_features.detach(), workspace_pool.detach()),
                                  dim=-1)).squeeze(-1)
