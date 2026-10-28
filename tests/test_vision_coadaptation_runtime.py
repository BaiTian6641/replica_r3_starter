"""Real Transformers 5.20.2 runtime tests for mutable co-adaptation."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab import vision_coadaptation as co  # noqa: E402


def _tiny_service():
    torch.manual_seed(7)
    config = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=12,
        num_attention_heads=2, patch_size=4, image_size=8)
    model = SiglipVisionModel(config)
    model.vision_projector = torch.nn.Sequential(
        torch.nn.LayerNorm(16), torch.nn.Linear(16, 32), torch.nn.GELU(),
        torch.nn.Linear(32, 16))
    spec = SimpleNamespace(
        encoder_width=16, max_visual_tokens=4, image_size=8, patch_size=4,
        digest="a" * 64)

    def processor(images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": torch.full((1, 3, 8, 8), 0.25)}

    service = co.MutableEncoderFeatureService.for_testing(
        model, spec=spec, processor=processor)
    shapes = {
        name: tuple(parameter.shape)
        for name, parameter in service.named_parameters()
        if name in co.SCOPE_TENSORS
    }
    report = service.enable_scope(tensor_shapes=shapes)
    return service, report



def _scope():
    return SimpleNamespace(subject_id="subject", branch_id="branch",
                           purpose="test")


def test_runtime_module_mapping_and_scope_resolution():
    service, report = _tiny_service()
    assert service.spec.encoder_width == 16
    assert service.spec.max_visual_tokens == 4
    assert report["runtime_to_checkpoint"]["vision_model.post_layernorm.weight"] == (
        "post_layernorm.weight")
    assert co.runtime_parameter_name(
        "vision_model.post_layernorm.bias") == "post_layernorm.bias"
    assert co.checkpoint_parameter_name(
        "post_layernorm.bias") == "vision_model.post_layernorm.bias"
    assert set(report["trainable_names"]) == set(co.SCOPE_TENSORS)
    assert all(not name.startswith("runtime.") for name in report)
    assert service._runtime_module("embeddings") is service._encoder.embeddings
    assert service._runtime_module("post_layernorm") is (
        service._encoder.post_layernorm)


def test_real_forward_backward_detachment_and_producer_invalidation(tmp_path):
    service, _ = _tiny_service()
    service._pixels = lambda path: (
        torch.full((1, 3, 8, 8), 0.25), "b" * 64)
    image = tmp_path / "tiny.png"
    observations = []
    handles = []
    for index in (0, 11):
        layer = service._encoder.encoder.layers[index]
        def record(module, args, output, index=index):
            observations.append((index, torch.is_grad_enabled()))
        handles.append(layer.register_forward_hook(record))

    try:
        first = service.observe(
            image, source_id="source", task_id="task", scope=_scope(),
            observed_at=0, source_epoch=0, training=False)
        assert not first.features.requires_grad
        assert observations == [(0, False), (11, False)]
        observations.clear()
        second = service.observe(
            image, source_id="source", task_id="task", scope=_scope(),
            observed_at=0, source_epoch=0, training=False)
        assert not second.features.requires_grad
        assert observations == []
        assert service._encoder_calls == 1

        trained = service.observe(
            image, source_id="source", task_id="task", scope=_scope(),
            observed_at=0, source_epoch=0, training=True)
        assert trained.features.requires_grad
        assert trained.features.grad_fn is not None
        assert observations == [(0, False), (11, True)]
        output = service._encoder.vision_projector(trained.features)
        loss = output.square().mean()
        loss.backward()
        gradients = co.scope_gradient_names(service.named_parameters())
        co.assert_gradient_ownership(gradients, expect_nonzero=True)
        assert set(gradients["present"]) == set(co.SCOPE_TENSORS)
        assert not gradients["unexpected"]
        assert not gradients["nonfinite"]

        before = service.producer_version()
        with torch.no_grad():
            service._encoder.post_layernorm.weight.add_(0.01)
        assert service.producer_version() != before
        service.observe(
            image, source_id="source", task_id="task", scope=_scope(),
            observed_at=0, source_epoch=0, training=False)
        audit = service.cache_audit()
        assert "producer_version_changed" in audit["invalidations"]
        assert audit["observation_identity"] is True
        assert service._encoder_calls == 3
    finally:
        for handle in handles:
            handle.remove()
