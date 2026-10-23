"""Persist one task-owned workspace between decisions (E3 §3).

Storage stays outside every forward/backward path: callers hand over detached
CPU tensors and this module only writes/reads files. A missing, stale or
incompatible payload is discarded (``load`` returns ``None``) and the exact
journal remains the only required copy of the task.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..architecture.contracts import CoreState

FORMAT = 'replica.task_workspace.v1'
_MAX_DIM = 8192


def detach_to_cpu(core: CoreState) -> CoreState:
    """The only inference-boundary copy; never shares storage with a live graph."""
    if core.workspace.device.type == 'cpu' and not core.workspace.requires_grad \
            and not core.valid.requires_grad and not core.task_features.requires_grad:
        return core
    return CoreState(core.workspace.detach().to('cpu'), core.valid.detach().to('cpu'),
                     core.task_features.detach().to('cpu'), core.producer, core.task_digest,
                     core.revision, core.steps, core.implementation)


class WorkspaceStore:
    """safetensors payload plus a validating JSON sidecar under one directory."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._weights = self.root / 'workspace.safetensors'
        self._sidecar = self.root / 'workspace.json'

    def save(self, core: CoreState) -> None:
        if not isinstance(core, CoreState):
            raise ValueError('WorkspaceStore persists a CoreState')
        if not isinstance(core.task_digest, str) or not core.task_digest:
            raise ValueError('WorkspaceStore requires the exact task digest binding')
        if not isinstance(core.producer, str) or not core.producer:
            raise ValueError('WorkspaceStore requires the producer identity')
        core = detach_to_cpu(core)
        slots, width, features = (int(core.workspace.shape[1]), int(core.workspace.shape[2]),
                                  int(core.task_features.shape[1]))
        if not 0 < slots <= _MAX_DIM or not 0 < width <= _MAX_DIM or not 0 < features <= _MAX_DIM:
            raise ValueError('Workspace dimensions exceed the bounded store size')
        save_file({'workspace': core.workspace.contiguous(), 'valid': core.valid.contiguous(),
                   'task_features': core.task_features.contiguous()}, str(self._weights))
        payload = {
            'format': FORMAT,
            'producer': core.producer,
            'task_digest': core.task_digest,
            'revision': core.revision,
            'steps': core.steps,
            'implementation': core.implementation,
            'slots': slots,
            'width': width,
            'features': features,
            'weights_sha256': hashlib.sha256(self._weights.read_bytes()).hexdigest(),
        }
        self._sidecar.write_text(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                            ensure_ascii=False) + '\n', encoding='utf-8')

    def load(self, *, expected_task_digest: str | None = None,
             expected_revision: int | None = None,
             expected_producer: str | None = None) -> CoreState | None:
        """Return the stored CoreState or None when missing/stale/incompatible.

        Optional bindings reject a payload produced by a different exact task,
        revision or producer; the caller rebuilds the workspace instead.
        """
        try:
            if not self._weights.is_file() or not self._sidecar.is_file():
                return None
            payload = json.loads(self._sidecar.read_text(encoding='utf-8'))
            if payload.get('format') != FORMAT:
                return None
            if hashlib.sha256(self._weights.read_bytes()).hexdigest() != payload.get('weights_sha256'):
                return None
            if expected_task_digest is not None \
                    and payload.get('task_digest') != expected_task_digest:
                return None
            if expected_revision is not None and payload.get('revision') != expected_revision:
                return None
            if expected_producer is not None and payload.get('producer') != expected_producer:
                return None
            if not isinstance(payload.get('producer'), str) or not payload['producer']:
                return None
            if not isinstance(payload.get('task_digest'), str) or not payload['task_digest']:
                return None
            if type(payload.get('revision')) is not int or payload['revision'] < 0:
                return None
            if type(payload.get('steps')) is not int or payload['steps'] < 0:
                return None
            if payload.get('implementation') not in ('recurrent', 'feedforward'):
                return None
            for name in ('slots', 'width', 'features'):
                if type(payload.get(name)) is not int or not 0 < payload[name] <= _MAX_DIM:
                    return None
            tensors = load_file(str(self._weights))
            if set(tensors) != {'workspace', 'valid', 'task_features'}:
                return None
            workspace = tensors['workspace']
            if (tuple(workspace.shape) != (1, payload['slots'], payload['width'])
                    or tuple(tensors['valid'].shape) != (1, payload['slots'])
                    or tuple(tensors['task_features'].shape) != (1, payload['features'])
                    or not workspace.is_floating_point()):
                return None
            return CoreState(workspace, tensors['valid'].to(torch.bool),
                             tensors['task_features'], payload['producer'],
                             payload['task_digest'], int(payload['revision']),
                             int(payload['steps']), payload['implementation'])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def clear(self) -> None:
        self._weights.unlink(missing_ok=True)
        self._sidecar.unlink(missing_ok=True)
