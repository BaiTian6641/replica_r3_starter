"""Causal loss accounting, schedule boundaries and prompt-only generation."""

import copy
import hashlib
import json

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, GenerationConfig

from replica_lab.training_tools import (apply_learning_rates, evaluate_generations,
                                        response_statistics)


def test_answer_eos_statistics_use_causal_positions_and_ignore_prompt():
    torch.manual_seed(3)
    logits = torch.randn(2, 5, 4, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 2, 0, -100, -100]])
    actual = response_statistics(logits, labels)
    log_probability = logits.detach().log_softmax(-1)
    expected_answer = -(log_probability[0, 1, 1] + log_probability[0, 2, 2]
                        + log_probability[1, 0, 2])
    expected_eos = -(log_probability[0, 3, 3] + log_probability[1, 1, 0])
    assert actual["answer_tokens"] == 3
    assert actual["eos_tokens"] == 2
    assert actual["answer_nll_sum"] == pytest.approx(expected_answer.item())
    assert actual["eos_nll_sum"] == pytest.approx(expected_eos.item())
    assert logits.grad is None
    changed = logits.detach().clone()
    changed[0, 0] = torch.tensor([900., -900., 0., 0.])
    changed[1, 2:] = -20
    changed[:, -1] = 30
    assert response_statistics(changed, labels) == actual
    changed[0, 3, 3] += 20
    eos_changed = response_statistics(changed, labels)
    assert eos_changed["answer_nll_sum"] == actual["answer_nll_sum"]
    assert eos_changed["eos_nll_sum"] < actual["eos_nll_sum"]
    changed[0, 1, 1] += 20
    answer_changed = response_statistics(changed, labels)
    assert answer_changed["eos_nll_sum"] == eos_changed["eos_nll_sum"]
    assert answer_changed["answer_nll_sum"] < eos_changed["answer_nll_sum"]


def test_statistics_empty_targets_and_first_token_are_excluded():
    logits = torch.zeros(2, 3, 4, requires_grad=True)
    labels = torch.tensor([[1, -100, -100], [-100, -100, -100]])
    assert response_statistics(logits, labels) == {
        "answer_nll_sum": 0.0, "answer_tokens": 0, "eos_nll_sum": 0.0, "eos_tokens": 0}


def scheduler_optimizer():
    return torch.optim.SGD([{"params": [nn.Parameter(torch.ones(1))], "lr": 0.1},
                            {"params": [nn.Parameter(torch.ones(1))], "lr": 0.01}])


def test_warmup_constant_and_cosine_have_explicit_boundaries():
    optimizer = scheduler_optimizer()
    base = [0.1, 0.01]
    constant = [apply_learning_rates(optimizer, base, step, 8, 2) for step in range(8)]
    assert constant == [0.5, 1., 1., 1., 1., 1., 1., 1.]
    cosine = [apply_learning_rates(optimizer, base, step, 8, 2, "cosine") for step in range(8)]
    assert cosine[:2] == [0.5, 1.]
    assert cosine[-1] == pytest.approx(0.2)
    assert all(first > second for first, second in zip(cosine[1:], cosine[2:]))
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.02, 0.002])
    assert base == [0.1, 0.01]
    # A saved next-step cursor and immutable rates reproduce a continuous run.
    assert apply_learning_rates(optimizer, base, 4, 8, 2, "cosine") == cosine[4]
    assert apply_learning_rates(optimizer, base, 0, 8, 0, "cosine") == 1.0
    assert apply_learning_rates(optimizer, base, 7, 8, 0, "cosine") == pytest.approx(0.2)
    assert apply_learning_rates(optimizer, base, 0, 1, 0, "cosine") == 1.0


@pytest.mark.parametrize("arguments", [
    {"completed_steps": -1}, {"completed_steps": 8}, {"completed_steps": True},
    {"total_steps": 0}, {"warmup_steps": 9}, {"schedule": "unknown"},
    {"min_lr_ratio": float("nan")}, {"base_rates": [0.1]}, {"base_rates": [-1, 0.1]},
])
def test_invalid_schedule_cannot_mutate_rates(arguments):
    optimizer = scheduler_optimizer()
    kwargs = {"base_rates": [0.1, 0.01], "completed_steps": 0,
              "total_steps": 8, "warmup_steps": 2, **arguments}
    with pytest.raises(ValueError):
        apply_learning_rates(optimizer, **kwargs)
    assert [group["lr"] for group in optimizer.param_groups] == [0.1, 0.01]


class FakeTokenizer:
    eos_token_id = 4
    tokens = {1: "0", 2: "1", 3: "The answer is 0", 4: "<eos>", 6: "00"}

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(self.tokens[token] for token in tokens)


class FakeBase(nn.Module):
    def __init__(self, outputs, fail=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.config = LlamaConfig(vocab_size=10, eos_token_id=4, bos_token_id=7,
                                   pad_token_id=4, use_cache=False)
        self.generation_config = GenerationConfig(do_sample=True, temperature=0.3)
        self.gradient_checkpointing = True
        self.outputs, self.calls, self.fail = outputs, [], fail

    def generate(self, **kwargs):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        assert set(kwargs) == {"input_ids", "attention_mask", "generation_config",
                               "do_sample", "use_cache", "max_new_tokens"}
        assert kwargs["do_sample"] is False
        assert kwargs["use_cache"] is True
        configuration = kwargs["generation_config"]
        assert configuration.do_sample is False
        assert configuration.num_beams == 1
        assert configuration.use_cache is True
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("generation failed")
        generated = torch.tensor([self.outputs[len(self.calls) - 1]])
        return torch.cat([kwargs["input_ids"], generated], dim=1)


class FakeAdapter(nn.Module):
    def __init__(self, outputs, fail=False):
        super().__init__()
        self.regime = "full"
        self.base = FakeBase(outputs, fail)


def encoded_rows(count):
    batches = [{"prompt_ids": torch.tensor([[7, 8, 9]]),
                "prompt_mask": torch.ones(1, 3, dtype=torch.bool),
                "input_ids": torch.tensor([[7, 8, 9, 1, 4]]),
                "labels": torch.tensor([[-100, -100, -100, 1, 4]])} for _ in range(count)]
    rows = [{"id": str(index), "response": "0"} for index in range(count)]
    return batches, rows


def test_generation_scores_whole_answers_and_never_passes_references():
    model = FakeAdapter([[1, 4], [3, 4], [6, 4], [2, 2]])
    model.train()
    # Retain an intentional child eval mode while the root is training.
    model.base.eval()
    original = copy.deepcopy(model.state_dict())
    batches, rows = encoded_rows(4)
    report = evaluate_generations(model, batches, rows, "cpu", 2, tokenizer=FakeTokenizer())
    assert report["task_accuracy"] == 0.5
    assert report["exact_text_accuracy"] == 0.25
    assert report["response_histogram"] == {"0": 1, "The answer is 0": 1, "00": 1, "11": 1}
    assert report["distinct_responses"] == 4
    assert report["examples"][1]["predicted_integer"] is None
    assert report["examples"][2]["strict_integer_match"] is True
    assert report["examples"][3]["hit_token_budget"] is True
    assert report["examples"][0]["ended_by_eos"] is True
    assert report["examples"][0]["raw_completion"] == "0<eos>"
    expected_hash = hashlib.sha256(json.dumps([[7, 8, 9]]).encode()).hexdigest()
    assert report["examples"][0]["prompt_token_ids_sha256"] == expected_hash
    for call in model.base.calls:
        assert torch.equal(call["input_ids"], batches[0]["prompt_ids"])
        assert call["input_ids"].shape[1] == 3
    assert model.training is True
    assert model.base.training is False
    assert model.base.config.use_cache is False
    assert model.base.gradient_checkpointing is True
    assert model.base.generation_config.do_sample is True
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_generation_restores_training_mode_after_exception():
    model = FakeAdapter([[1, 4]], fail=True).train()
    batches, rows = encoded_rows(1)
    with pytest.raises(RuntimeError, match="generation failed"):
        evaluate_generations(model, batches, rows, "cpu", tokenizer=FakeTokenizer())
    assert model.training is True
    assert model.base.training is True


def test_augmented_generation_receives_only_prompt_and_source(monkeypatch):
    model = FakeAdapter([[1, 4]])
    model.regime = "full_augmented"
    batches, rows = encoded_rows(1)
    batches[0].update(memory_ids=torch.tensor([[5, 6]]),
                      memory_mask=torch.ones(1, 2, dtype=torch.bool))
    calls = []
    def generated(adapter, **kwargs):
        assert adapter is model
        assert set(kwargs) == {"prompt_ids", "prompt_mask", "memory_ids", "memory_mask",
                               "max_new_tokens", "eos_token_id"}
        assert torch.equal(kwargs["prompt_ids"], batches[0]["prompt_ids"])
        assert torch.equal(kwargs["memory_ids"], batches[0]["memory_ids"])
        calls.append(kwargs)
        return torch.tensor([[7, 8, 9, 1, 4]])
    monkeypatch.setattr("replica_lab.generation.generate_fixed_context", generated)
    report = evaluate_generations(model, batches, rows, "cpu", tokenizer=FakeTokenizer())
    assert report["task_accuracy"] == 1
    assert report["decoding"]["use_cache"] is False
    assert len(calls) == 1
    assert not model.base.calls
