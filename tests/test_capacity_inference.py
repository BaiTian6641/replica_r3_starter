"""Regression guard for inference-only capacity producers."""
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _capacity_script():
    spec = importlib.util.spec_from_file_location(
        'run_e2_capacity', ROOT / 'scripts' / 'run_e2_capacity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inference_call_does_not_retain_autograd():
    module = _capacity_script()
    model = torch.nn.Linear(4, 4)
    inputs = torch.randn(1, 4, requires_grad=True)
    with_grad = model(inputs)
    without_grad = module.inference_call(torch, lambda: model(inputs))
    assert with_grad.grad_fn is not None
    assert without_grad.grad_fn is None
    assert without_grad.requires_grad is False
