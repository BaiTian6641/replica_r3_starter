"""Action-kind head: batched shape, finite logits and detached inputs."""

import torch

from replica_lab.architecture.contracts import CoreSpec
from replica_lab.architecture.heads import ACTION_KINDS, ActionKindHead

SPEC = CoreSpec(width=16, slots=2, steps=1, heads=2, task_features=4,
                ffn_hidden=32, control_hidden=8)


def test_action_kind_head_batch_and_detached_inputs():
    head = ActionKindHead(SPEC, hidden=8)
    task = torch.randn(3, SPEC.task_features, requires_grad=True)
    workspace = torch.randn(3, SPEC.slots, SPEC.width, requires_grad=True)
    valid = torch.ones(3, SPEC.slots, dtype=torch.bool)
    logits = head(task, workspace, valid)
    assert logits.shape == (3, len(ACTION_KINDS))
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert head.mlp[0].weight.grad is not None
    assert torch.isfinite(head.mlp[0].weight.grad).all()
    assert task.grad is None and workspace.grad is None


def test_action_kind_head_mask_and_shape_contracts():
    head = ActionKindHead(SPEC, hidden=8)
    task = torch.randn(1, SPEC.task_features)
    workspace = torch.randn(1, SPEC.slots, SPEC.width)
    valid = torch.tensor([[True, False]])
    assert head(task, workspace, valid).shape == (1, len(ACTION_KINDS))
    try:
        head(task, workspace[:, :1], valid)
    except ValueError:
        pass
    else:
        raise AssertionError('slot mismatch must fail')
    try:
        head(task, workspace, torch.ones(1, 3, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError('mask mismatch must fail')
