"""Shared helpers for the v12 CPU engineering tests; no native claims."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import torch

from replica_lab.contextual_v12_fixture import (
    ENGINEERING_LABEL, SyntheticTokenizer, SyntheticWorkAccounting, build_v12_stack,
    canonical_span, canonical_spans, logits_digest, model_state_digest, open_v12_store,
    optimizer_digest, parameter_digest, scope, source_feature_spec, spans_for)
from replica_lab.tiered_memory import MemoryScope

ROOT = Path(__file__).resolve().parents[1]


def token_inputs(sequence=(3, 5, 7, 9, 11, 13)):
    ids = torch.tensor([list(sequence)], dtype=torch.long)
    mask = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, 0] = -100
    return ids, mask, labels


def step(path, adapter, spans, *, ids, labels, sc, active_sites=None,
         retain_graph=False):
    path.optimizer.zero_grad(set_to_none=True)
    batch = path.training_values(spans, scope=sc)
    batch.slots.retain_grad()
    logits = path.forward(ids, torch.ones_like(ids), batch, scope=sc,
                          active_sites=active_sites)
    loss = adapter.causal_loss(logits, ids, labels)
    loss.backward(retain_graph=retain_graph)
    path.optimizer.step()
    return batch, logits, loss


def backward_only(path, adapter, spans, *, ids, labels, sc, active_sites=None):
    path.optimizer.zero_grad(set_to_none=True)
    batch = path.training_values(spans, scope=sc)
    batch.slots.retain_grad()
    logits = path.forward(ids, torch.ones_like(ids), batch, scope=sc,
                          active_sites=active_sites)
    loss = adapter.causal_loss(logits, ids, labels)
    loss.backward()
    return batch, logits, loss


def grad_state(module):
    from collections import Counter
    states = []
    for _, parameter in module.named_parameters():
        if parameter.grad is None:
            states.append("none")
        elif not bool(torch.count_nonzero(parameter.grad)):
            states.append("zero")
        else:
            states.append("nonzero")
    return Counter(states)


def grads_are_finite(module):
    for _, parameter in module.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            return False
    return True


def nonzero_gradient_names(module):
    return {name for name, parameter in module.named_parameters()
            if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))}


def run_child(code, root, *, timeout=300):
    return subprocess.run(
        [sys.executable, "-c", code, str(root)], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src:tests", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=timeout)


__all__ = [
    "ENGINEERING_LABEL", "ROOT", "SyntheticTokenizer", "SyntheticWorkAccounting",
    "backward_only", "build_v12_stack", "canonical_span", "canonical_spans",
    "grad_state", "grads_are_finite", "logits_digest", "model_state_digest",
    "nonzero_gradient_names", "open_v12_store", "optimizer_digest", "parameter_digest",
    "run_child", "scope", "source_feature_spec", "spans_for", "step", "token_inputs",
    "MemoryScope",
]
