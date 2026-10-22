"""Offline tiny native-Llama mechanism tests; no checkpoint downloads."""
import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from replica_lab.model_adapter import LlamaMemoryAdapter


def tiny(regime="modules_only"):
    torch.manual_seed(9)
    config = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=64,
                         attention_dropout=0.0, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    return LlamaMemoryAdapter(LlamaForCausalLM(config), regime, memory_size=16,
                              workspace_size=16, slots=2, workspace_steps=1)


def batch():
    return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "labels": torch.tensor([[-100, -100, -100, 4, 5]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "prompt_ids": torch.tensor([[1, 2, 3]]),
            "memory_ids": torch.tensor([[7, 8]]),
            "prompt_mask": torch.ones(1, 3, dtype=torch.bool),
            "memory_mask": torch.ones(1, 2, dtype=torch.bool)}


@pytest.mark.parametrize("regime", ["baseline", "full", "modules_only", "full_augmented"])
def test_exact_optimizer_scope(regime):
    model = tiny(regime)
    groups = model.optimizer_groups(1e-4, 1e-3)
    got = {id(p) for g in groups for p in g["params"]}
    assert got == {id(p) for p in model.parameters() if p.requires_grad}
    audit = model.parameter_audit()
    assert (audit["original"]["trainable"] == audit["original"]["total"]) == (regime in {"full", "full_augmented"})
    if regime == "baseline":
        assert not got


def test_zero_initialized_residual_preserves_native_logits():
    model = tiny().eval()
    b = batch()
    original = model.base.model.layers[-1].original
    with torch.no_grad():
        output = model(**b).logits
        model.release_context()
        wrapped = model.base.model.layers[-1]
        model.base.model.layers[-1] = original
        native = model.base(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits
        model.base.model.layers[-1] = wrapped
    torch.testing.assert_close(output, native, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_writer_gradients_and_frozen_base(checkpointing):
    model = tiny()
    if checkpointing:
        model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    before = {n: p.detach().clone() for n, p in model._original_parameters.items()}
    optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 1e-2), foreach=False)
    writer_has_gradient = False
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        result = model(**batch())
        result.loss.backward()
        writer_has_gradient |= any(p.grad is not None and bool(torch.count_nonzero(p.grad))
                                   for p in model.support_writer.parameters())
        model.release_context()
        optimizer.step()
    assert writer_has_gradient  # may open only after zero-init output projections learn
    for name, parameter in model._original_parameters.items():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_prompt_context_does_not_read_target_tokens():
    model = tiny().eval()
    b = batch()
    model(**b)
    first = tuple(t.detach().clone() for t in model._context)
    model.release_context()
    b["input_ids"] = torch.tensor([[1, 2, 3, 22, 23]])
    model(**b)
    second = model._context
    for a, c in zip(first, second):
        torch.testing.assert_close(a, c, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="still in flight"):
        model(**b)
    model.release_context()


def test_native_base_and_sidecar_roundtrip(tmp_path):
    model = tiny("full_augmented").eval()
    with torch.no_grad():
        model.memory_reader.output_projection.weight.normal_(std=0.02)
        model.workspace.reader.output_projection.weight.normal_(std=0.02)
        expected = model(**batch()).logits
        model.release_context()
    model.save_pretrained(tmp_path)
    base = LlamaForCausalLM.from_pretrained(tmp_path / "base", local_files_only=True)
    loaded = LlamaMemoryAdapter(base, "full_augmented", memory_size=16,
                                workspace_size=16, slots=2, workspace_steps=1).eval()
    loaded.load_sidecars(tmp_path / "sidecars.safetensors")
    with torch.no_grad():
        actual = loaded(**batch()).logits
        loaded.release_context()
    torch.testing.assert_close(actual, expected)


def test_full_updates_embeddings_and_output_head():
    model = tiny("full")
    embeddings = model.base.get_input_embeddings().weight
    head = model.base.get_output_embeddings().weight
    before_embeddings, before_head = embeddings.detach().clone(), head.detach().clone()
    optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 1e-3), foreach=False)
    model(**batch()).loss.backward()
    assert not model.gradient_audit()["original"]["missing_gradient_names"]
    optimizer.step()
    model.release_context()
    assert not torch.equal(embeddings, before_embeddings)
    assert not torch.equal(head, before_head)


def test_prompt_target_contamination_rejected():
    model = tiny()
    data = batch()
    data["prompt_ids"] = data["input_ids"]
    data["prompt_mask"] = torch.ones_like(data["input_ids"], dtype=torch.bool)
    with pytest.raises(ValueError, match="supervised target"):
        model(**data)


def test_completion_mask_and_shared_evidence():
    spec = importlib.util.spec_from_file_location("train_text", Path(__file__).parents[1] / "scripts/train_text.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class FakeTokenizer:
        eos_token_id = 47
        def apply_chat_template(self, messages, **kwargs):
            assert "observed-marker" in messages[0]["content"]
            return "prefix"
        def encode(self, text, **kwargs):
            return [10 + len(text) % 10, 11]
    result = module.encode_row({"id": "a", "prompt": "question",
                                "response": "answer", "memory_texts": ["observed-marker"]},
                               FakeTokenizer(), 20)
    assert result["labels"].tolist() == [[-100, -100, 16, 11, 47]]
    assert result["prompt_ids"].tolist() == [[16, 11]]
    assert result["memory_ids"].tolist() == [[15, 11]]
