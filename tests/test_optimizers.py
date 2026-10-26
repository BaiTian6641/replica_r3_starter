"""Numerical, routing, state compression, and restart contracts for Muon8bit."""

import copy
import io

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from replica_lab.model_adapter import LlamaMemoryAdapter
from replica_lab.optimizers import (NativeFusedAdamW, build_optimizer,
                                    dequantize_blockwise, orthogonalize_fp32,
                                    quantize_blockwise)


class MatrixModel(nn.Module):
    def __init__(self, shape=(37, 71)):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(shape) * 0.02)
        self.bias = nn.Parameter(torch.zeros(shape[0]))

    def optimizer_groups(self, base_lr, new_lr):
        return [{"params": list(self.parameters()), "lr": base_lr, "group_name": "original"}]


def tiny(regime="full_augmented", tied=False):
    torch.manual_seed(19)
    config = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=64,
                         attention_dropout=0.0, tie_word_embeddings=tied)
    config._attn_implementation = "eager"
    return LlamaMemoryAdapter(LlamaForCausalLM(config), regime, memory_size=16,
                              workspace_size=16, slots=2, workspace_steps=1)


def batch(device="cpu"):
    return {key: value.to(device) for key, value in {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[-100, -100, -100, 4, 5]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "prompt_ids": torch.tensor([[1, 2, 3]]),
        "memory_ids": torch.tensor([[7, 8]]),
        "prompt_mask": torch.ones(1, 3, dtype=torch.bool),
        "memory_mask": torch.ones(1, 2, dtype=torch.bool)}.items()}


def test_quantization_error_zero_and_partial_blocks():
    torch.manual_seed(1)
    value = torch.randn(13, 401)
    value.reshape(-1)[:2048] = 0
    codes, scales = quantize_blockwise(value)
    actual = dequantize_blockwise(codes, scales)
    error_bound = scales.repeat_interleave(2048)[:value.numel()].view_as(value) / 2
    assert codes.dtype == torch.int8 and scales.dtype == torch.float32
    assert codes.untyped_storage().nbytes() == value.numel()
    assert torch.all((actual - value).abs() <= error_bound + 2e-7)
    assert torch.count_nonzero(actual.reshape(-1)[:2048]) == 0
    assert codes.numel() + 4 * scales.numel() < value.numel() * 1.01


def test_muon_matches_fp32_reference_before_quantization_error():
    torch.manual_seed(3)
    model = MatrixModel()
    optimizer = build_optimizer(model, lr_muon_base=0.003, weight_decay=0.02)
    expected = model.matrix.detach().clone()
    gradient = torch.randn_like(model.matrix)
    model.matrix.grad = gradient
    # First momentum is exactly the current gradient, so quantization only
    # affects the next update. SVD is deliberately not the NS approximation.
    update = orthogonalize_fp32(gradient)
    expected.mul_(1 - 0.003 * 0.02).add_(update, alpha=-0.003 * 0.2 * max(expected.shape)**0.5)
    optimizer.step()
    torch.testing.assert_close(model.matrix, expected, atol=1e-8, rtol=1e-6)
    state = optimizer.state[model.matrix]
    assert set(state) == {"momentum_q8", "momentum_scale"}
    assert state["momentum_q8"].dtype == torch.int8
    # A real second update uses the saved dequantized state.
    gradient2 = torch.randn_like(model.matrix)
    momentum2 = 0.95 * dequantize_blockwise(state["momentum_q8"], state["momentum_scale"]) + gradient2
    expected.mul_(1 - 0.003 * 0.02).add_(orthogonalize_fp32(momentum2),
                                       alpha=-0.003 * 0.2 * max(expected.shape)**0.5)
    model.matrix.grad = gradient2
    optimizer.step()
    torch.testing.assert_close(model.matrix, expected, atol=1e-8, rtol=1e-6)


def test_adamw_matches_pytorch_multiple_updates():
    torch.manual_seed(6)
    model = MatrixModel()
    other = copy.deepcopy(model)
    actual = build_optimizer(model, "adamw", lr_base=0.002, weight_decay=0.03)
    reference = torch.optim.AdamW(other.parameters(), lr=0.002, weight_decay=0.03, foreach=False)
    for _ in range(5):
        for p, q in zip(model.parameters(), other.parameters()):
            gradient = torch.randn_like(p)
            p.grad, q.grad = gradient, gradient.clone()
        actual.step()
        reference.step()
    for p, q in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p, q, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("shape", [(37, 71), (128, 48)])
def test_ns_direction_agrees_with_independent_svd_polar(shape):
    torch.manual_seed(31)
    gradient = torch.randn(shape)
    left, _, right = torch.linalg.svd(gradient, full_matrices=False)
    polar = left @ right
    actual = orthogonalize_fp32(gradient)
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), polar.flatten(), dim=0)
    # The quintic five-step approximation intentionally does not return an
    # exact polar factor. Check direction against an independent SVD method.
    assert cosine > 0.97
    torch.testing.assert_close(orthogonalize_fp32(torch.zeros(shape)), torch.zeros(shape))


def test_first_muon_update_agrees_with_stock_muon_bf16_ns_tolerance():
    torch.manual_seed(31)
    model = MatrixModel((128, 48))
    reference_parameter = nn.Parameter(model.matrix.detach().clone())
    actual = build_optimizer(model, lr_muon_base=0.003, weight_decay=0)
    reference = torch.optim.Muon([reference_parameter], lr=0.003, weight_decay=0,
                                 nesterov=False, adjust_lr_fn="match_rms_adamw")
    gradient = torch.randn_like(model.matrix)
    model.matrix.grad, reference_parameter.grad = gradient, gradient.clone()
    before = model.matrix.detach().clone()
    actual.step()
    reference.step()
    update, stock_update = before - model.matrix, before - reference_parameter
    # Stock torch 2.14 Muon uses BF16 inside NS; the portable implementation
    # deliberately keeps FP32 arithmetic on every platform.
    assert (update - stock_update).norm() / update.norm() < 0.03


@pytest.mark.parametrize("regime", ["baseline", "full", "modules_only", "full_augmented"])
@pytest.mark.parametrize("tied", [False, True])
def test_exact_scope_and_embedding_head_routing(regime, tied):
    model = tiny(regime, tied)
    optimizer = build_optimizer(model, lr_base=1e-5, lr_new=2e-4,
                                lr_muon_base=3e-3, lr_muon_new=4e-3)
    if regime == "baseline":
        assert optimizer is None
        return
    members = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(members) == len({id(p) for p in members})
    assert {id(p) for p in members} == {id(p) for p in model.parameters() if p.requires_grad}
    excluded = {id(model.base.get_input_embeddings().weight), id(model.base.get_output_embeddings().weight)}
    excluded.update(id(p) for name, p in model.named_parameters()
                    if name.endswith("slot_queries"))
    for group in optimizer.param_groups:
        if group["algorithm"] == "muon8bit":
            assert all(p.ndim == 2 and id(p) not in excluded for p in group["params"])
            assert group["lr"] == (3e-3 if group["scope"] == "original" else 4e-3)
        else:
            assert group["lr"] == (1e-5 if group["scope"] == "original" else 2e-4)
    adamw_ids = {id(p) for group in optimizer.param_groups if group["algorithm"] == "adamw"
                 for p in group["params"]}
    assert excluded.intersection({id(p) for p in members}).issubset(adamw_ids)


def test_native_llama_updates_base_embeddings_head_and_new_modules():
    model = tiny()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = build_optimizer(model, lr_base=1e-3, lr_new=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(**batch()).loss.backward()
        model.release_context()
        optimizer.step()
    assert not model.gradient_audit()["original"]["missing_gradient_names"]
    assert not model.gradient_audit()["new"]["missing_gradient_names"]
    for prefix in ("base.model.embed_tokens", "base.lm_head", "support_writer", "memory_reader", "workspace"):
        assert any(not torch.equal(p, before[name]) for name, p in model.named_parameters()
                   if name.startswith(prefix))
    audit = optimizer.audit()
    assert audit["state_bytes_by_dtype"]["torch.int8"] > 0
    assert audit["state_tensor_bytes"] < sum(p.numel() * 8 for p in model.parameters())


def test_checkpoint_preserves_int8_and_restart_equivalence():
    torch.manual_seed(4)
    model = MatrixModel()
    optimizer = build_optimizer(model)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    optimizer.step()
    stream = io.BytesIO()
    torch.save(optimizer.state_dict(), stream)
    stream.seek(0)
    loaded = copy.deepcopy(model)
    restored = build_optimizer(loaded)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert restored.state[loaded.matrix]["momentum_q8"].dtype == torch.int8
    assert restored.audit() == optimizer.audit()
    for p, q in zip(model.parameters(), loaded.parameters()):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
    optimizer.step()
    restored.step()
    for p, q in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_rms_aligned_default_rates_follow_each_scope():
    model = tiny()
    optimizer = build_optimizer(model, lr_base=2e-5, lr_new=7e-5)
    for group in optimizer.param_groups:
        assert group["lr"] == (2e-5 if group["scope"] == "original" else 7e-5)


def test_rejects_bad_precision_gradient_and_checkpoint_scope():
    with pytest.raises(ValueError, match="FP32"):
        build_optimizer(MatrixModel().half())
    model = MatrixModel()
    optimizer = build_optimizer(model)
    before = model.matrix.detach().clone()
    model.matrix.grad = torch.ones_like(model.matrix)
    model.bias.grad = torch.full_like(model.bias, float("nan"))
    with pytest.raises(ValueError, match="finite dense"):
        optimizer.step()
    torch.testing.assert_close(model.matrix, before, atol=0, rtol=0)
    saved = optimizer.state_dict()
    saved["param_groups"][0]["param_names"] = ["wrong"]
    with pytest.raises(ValueError, match="param_names"):
        optimizer.load_state_dict(saved)


def test_native_fused_matches_direct_torch_and_records_actual_state():
    torch.manual_seed(61)
    model = MatrixModel()
    other = copy.deepcopy(model)
    optimizer = build_optimizer(model, "adamw_fused", lr_base=0.002,
                                weight_decay=0.03)
    assert isinstance(optimizer, NativeFusedAdamW)
    reference = torch.optim.AdamW(other.parameters(), lr=0.002,
                                  weight_decay=0.03, fused=True)
    for _ in range(3):
        for parameter, expected in zip(model.parameters(), other.parameters()):
            gradient = torch.randn_like(parameter)
            parameter.grad = gradient
            expected.grad = gradient.clone()
        optimizer.step()
        reference.step()
    for parameter, expected in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    audit = optimizer.audit()
    assert audit["optimizer_class"] == "torch.optim.AdamW"
    assert audit["implementation"] == "native_fused"
    assert audit["fused"] is True
    assert audit["library_version"] == str(torch.__version__)
    assert audit["state_dtypes"] == ["torch.float32"]
    assert audit["state_tensor_bytes"] > 0
    assert [(group["scope"], group["algorithm"], group["fused"])
            for group in audit["groups"]] == [("original", "adamw_fused", True)]


def test_native_fused_checkpoint_roundtrip_and_legacy_rejection():
    torch.manual_seed(62)
    model = MatrixModel()
    optimizer = build_optimizer(model, "adamw_fused", lr_base=0.002)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    stream = io.BytesIO()
    torch.save(optimizer.state_dict(), stream)
    stream.seek(0)
    restored_model = copy.deepcopy(model)
    restored = build_optimizer(restored_model, "adamw_fused", lr_base=0.002)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert restored.audit() == optimizer.audit()
    for parameter, expected in zip(model.parameters(), restored_model.parameters()):
        gradient = torch.randn_like(parameter)
        parameter.grad = gradient
        expected.grad = gradient.clone()
    optimizer.step()
    restored.step()
    for parameter, expected in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

    legacy = torch.optim.AdamW(model.parameters(), fused=True).state_dict()
    before = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError, match="Incompatible native fused"):
        optimizer.load_state_dict(legacy)
    assert_nested = optimizer.state_dict()
    assert_nested["state"].keys() == before["state"].keys()
    assert_nested["param_groups"] == before["param_groups"]
    custom = build_optimizer(MatrixModel(), "adamw").state_dict()
    with pytest.raises(ValueError, match="Incompatible native fused"):
        restored.load_state_dict(custom)


def test_native_fused_rejects_missing_duplicate_and_nonfinite_ownership():
    class DuplicateModel(MatrixModel):
        def optimizer_groups(self, base_lr, new_lr):
            return [{"params": [self.matrix], "lr": base_lr, "group_name": "original"},
                    {"params": [self.matrix], "lr": base_lr, "group_name": "original"}]

    class MissingModel(MatrixModel):
        def optimizer_groups(self, base_lr, new_lr):
            return [{"params": [self.matrix], "lr": base_lr, "group_name": "original"}]

    with pytest.raises(RuntimeError, match="incomplete or duplicated"):
        build_optimizer(DuplicateModel(), "adamw_fused")
    with pytest.raises(RuntimeError, match="incomplete or duplicated"):
        build_optimizer(MissingModel(), "adamw_fused")
    with pytest.raises(ValueError, match="FP32"):
        build_optimizer(MatrixModel().half(), "adamw_fused")

    model = MatrixModel()
    optimizer = build_optimizer(model, "adamw_fused")
    before = model.matrix.detach().clone()
    model.matrix.grad = torch.full_like(model.matrix, float("nan"))
    with pytest.raises(ValueError, match="finite dense"):
        optimizer.step()
    torch.testing.assert_close(model.matrix, before, rtol=0, atol=0)
    model.matrix.grad = None
    model.bias.grad = torch.sparse_coo_tensor(
        torch.tensor([[0]]), torch.tensor([1.0]), model.bias.shape)
    with pytest.raises(ValueError, match="sparse"):
        optimizer.step()
    torch.testing.assert_close(model.matrix, before, rtol=0, atol=0)


@pytest.mark.parametrize("regime", ["baseline", "full", "modules_only", "full_augmented"])
def test_native_fused_preserves_declared_optimizer_scope(regime):
    model = tiny(regime)
    optimizer = build_optimizer(model, "adamw_fused", lr_base=1e-5, lr_new=2e-4)
    if regime == "baseline":
        assert optimizer is None
        return
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(owned) == len({id(parameter) for parameter in owned})
    assert {id(parameter) for parameter in owned} == {id(parameter) for parameter in expected}
    assert all(group["algorithm"] == "adamw_fused" and group["fused"] is True
               for group in optimizer.param_groups)
    if regime in {"full", "full_augmented"}:
        names = {name for group in optimizer.param_groups for name in group["param_names"]}
        assert any("embed_tokens" in name for name in names)
        assert any("lm_head" in name for name in names)
        assert any(name.endswith("model.norm.weight") for name in names)
