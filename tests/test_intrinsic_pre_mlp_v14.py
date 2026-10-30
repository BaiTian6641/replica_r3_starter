"""Pre-MLP fusion placement mechanics on the tiny CPU backbone; no quality claim."""
import pytest
import torch

from replica_lab.architecture.intrinsic_memory_layout_v4 import PRE_MLP_PLACEMENT
from replica_lab.contextual_memory import ContextualMemoryError
from replica_lab.contextual_v12_fixture import (
    build_v12_stack, canonical_span, logits_digest, scope)


def ids():
    return torch.tensor([[3, 5, 7, 9, 11, 13]], dtype=torch.long)


def path_for(placement, execution="CPU_pre_mlp_test"):
    _, _, _, path = build_v12_stack(
        seed=17, layers=8, sites=(3, 7), slots=8, vocab_size=64, lr=1e-3,
        initialization_seed=17, cache_entries=16,
        source_identity="pre_mlp_cpu_fixture", execution_identity=execution,
        objective_identity={"scope": "pre_mlp_mechanics"}, placement=placement)
    return path


def test_pre_mlp_null_parity_influence_and_gradients():
    torch.set_num_threads(1)
    post = path_for("block_input_query_to_post_block_residual", "post")
    pre = path_for(PRE_MLP_PLACEMENT, "pre")
    sc = scope("pre_mlp")
    spans = canonical_span("the badge is red", record_id="badge")
    inputs = ids()
    mask = torch.ones_like(inputs)
    labels = inputs.clone()
    labels[:, 0] = -100
    try:
        for path in (post, pre):
            batch = path.training_values(spans, scope=sc)
            with torch.no_grad():
                disabled = path.forward(inputs, mask, batch, scope=sc, enabled=False)
                enabled = path.forward(inputs, mask, batch, scope=sc)
            assert torch.equal(disabled, enabled), path.spec.placement
        base = logits_digest(disabled)
        for _ in range(2):
            pre.optimizer.zero_grad(set_to_none=True)
            batch = pre.training_values(spans, scope=sc)
            logits = pre.forward(inputs, mask, batch, scope=sc)
            loss = pre.adapter.causal_loss(logits, inputs, labels)
            loss.backward()
            pre.optimizer.step()
        assert logits_digest(logits) != base
        batch = pre.training_values(spans, scope=sc)
        pre.optimizer.zero_grad(set_to_none=True)
        logits = pre.forward(inputs, mask, batch, scope=sc)
        loss = pre.adapter.causal_loss(logits, inputs, labels)
        loss.backward()
        assert float(pre.reader.ports["3"].output_projection.weight.grad.norm()) > 0
        assert float(pre.encoder.input_projection.weight.grad.norm()) > 0
        context = pre.snapshot(inputs, batch, scope=sc)
        _, audit = pre.adapter.forward_text(
            inputs, mask, use_projector=False, use_ports=False, use_spine=False,
            intrinsic_memory=context, intrinsic_placement=PRE_MLP_PLACEMENT,
            return_audit=True)
        assert set(audit.injected_layers) == {3, 7}
        assert audit.delta_max_abs > 0
    finally:
        post.close()
        pre.close()


def test_pre_mlp_checkpoint_placement_identity_is_explicit(tmp_path):
    torch.set_num_threads(1)
    pre = path_for(PRE_MLP_PLACEMENT, "same_execution")
    try:
        pre.save(tmp_path / "checkpoint")
    finally:
        pre.close()
    post = path_for("block_input_query_to_post_block_residual", "same_execution")
    try:
        with pytest.raises(ContextualMemoryError, match="identity differs"):
            post.load(tmp_path / "checkpoint")
    finally:
        post.close()
    restored = path_for(PRE_MLP_PLACEMENT, "same_execution")
    try:
        restored.load(tmp_path / "checkpoint")
        assert restored.spec.placement == PRE_MLP_PLACEMENT
    finally:
        restored.close()
