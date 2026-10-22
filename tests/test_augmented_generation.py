"""CPU tiny-native greedy/context contracts; no downloads or model-quality claim."""

import copy

import pytest
import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

from replica_lab.generation import generate_fixed_context
from replica_lab.model_adapter import LlamaMemoryAdapter


def tiny(regime="modules_only"):
    torch.manual_seed(83)
    config = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=64,
                         attention_dropout=0.0, tie_word_embeddings=False,
                         bos_token_id=1, eos_token_id=47, pad_token_id=0)
    config._attn_implementation = "eager"
    return LlamaMemoryAdapter(LlamaForCausalLM(config), regime, memory_size=16,
                              workspace_size=16, slots=2, workspace_steps=2)


def inputs(left_padding=False):
    prompt = [[0, 0, 1, 2, 3]] if left_padding else [[1, 2, 3]]
    return {"prompt_ids": torch.tensor(prompt),
            "prompt_mask": torch.tensor([[0, 0, 1, 1, 1]] if left_padding else [[1, 1, 1]],
                                         dtype=torch.bool),
            "memory_ids": torch.tensor([[7, 8]]),
            "memory_mask": torch.ones(1, 2, dtype=torch.bool)}


@pytest.mark.parametrize("regime", ["baseline", "full", "modules_only", "full_augmented"])
@pytest.mark.parametrize("left_padding", [False, True])
def test_null_branch_tokens_match_native_uncached_generate(regime, left_padding):
    model = tiny(regime)
    values = inputs(left_padding)
    # A native copy is made before wrapping/unwrapping; both share exact weights.
    native = LlamaForCausalLM(copy.deepcopy(model.base.config))
    if model.augmented:
        native.load_state_dict({name: p for name, p in model._original_parameters.items()})
    else:
        native.load_state_dict(model.base.state_dict())
    native.eval()
    configuration = GenerationConfig.from_model_config(native.config)
    configuration.do_sample = False
    configuration.num_beams = 1
    configuration.use_cache = False
    configuration.eos_token_id = []
    with torch.inference_mode():
        expected = native.generate(input_ids=values["prompt_ids"],
                                   attention_mask=values["prompt_mask"],
                                   generation_config=configuration, max_new_tokens=4)
    actual = generate_fixed_context(model, **values, max_new_tokens=4, eos_token_id=[])
    assert torch.equal(actual, expected)
    assert model._context is None


def test_nonzero_sidecar_changes_greedy_answers():
    model = tiny()
    original = generate_fixed_context(model, **inputs(), max_new_tokens=4, eos_token_id=[])
    with torch.no_grad():
        model.workspace.reader.output_projection.weight.normal_(std=2)
    changed = generate_fixed_context(model, **inputs(), max_new_tokens=4, eos_token_id=[])
    assert not torch.equal(changed[:, 3:], original[:, 3:])
    assert torch.equal(changed[:, :3], inputs()["prompt_ids"])


def test_context_builds_once_from_original_inputs_and_never_answers():
    model = tiny()
    values = inputs()
    counts = {"support": 0, "workspace": 0}
    context_ids, contexts, prefix_lengths = [], [], []
    expected_support = model.base.get_input_embeddings()(values["memory_ids"]).detach()
    expected_prompt = model.base.get_input_embeddings()(values["prompt_ids"]).detach()

    def source_hook(module, args):
        counts["support"] += 1
        torch.testing.assert_close(args[0], expected_support, rtol=0, atol=0)

    def workspace_hook(module, args):
        counts["workspace"] += 1
        torch.testing.assert_close(args[0], expected_prompt, rtol=0, atol=0)

    def base_hook(module, args, kwargs):
        assert "labels" not in kwargs
        assert kwargs["use_cache"] is False
        assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
        assert not model.training and not module.training
        prefix_lengths.append(kwargs["input_ids"].shape[1])
        context_ids.append(id(model._context))
        contexts.append(tuple(value.clone() for value in model._context))

    handles = [model.support_writer.register_forward_pre_hook(source_hook),
               model.workspace.encoder.register_forward_pre_hook(workspace_hook),
               model.base.register_forward_pre_hook(base_hook, with_kwargs=True)]
    try:
        result = generate_fixed_context(model, **values, max_new_tokens=4, eos_token_id=[])
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"support": 1, "workspace": 1}
    assert prefix_lengths == [3, 4, 5, 6]
    assert len(set(context_ids)) == 1
    for current in contexts[1:]:
        for first, next_value in zip(contexts[0], current):
            torch.testing.assert_close(first, next_value, rtol=0, atol=0)
    assert result.shape == (1, 7)
    assert model._context is None
    # There is no target-bearing argument in this inference interface.
    with pytest.raises(TypeError, match="labels"):
        generate_fixed_context(model, **values, labels=torch.tensor([[45]]))


def test_eos_integer_list_native_default_and_explicit_no_stop():
    model = tiny()
    values = inputs()
    unbounded = generate_fixed_context(model, **values, max_new_tokens=3, eos_token_id=[])
    first = unbounded[0, 3].item()
    for eos in (first, [first, (first + 1) % 48]):
        actual = generate_fixed_context(model, **values, max_new_tokens=3, eos_token_id=eos)
        assert actual.shape == (1, 4)
        assert actual[0, -1].item() == first
    model.base.config.eos_token_id = first
    assert generate_fixed_context(model, **values, max_new_tokens=3).shape == (1, 4)
    assert generate_fixed_context(model, **values, max_new_tokens=3,
                                  eos_token_id=[]).shape == (1, 6)
    assert model.base.config.eos_token_id == first


def test_empty_memory_is_supported_without_rebuilding_context():
    model = tiny()
    values = inputs()
    values.pop("memory_ids")
    values.pop("memory_mask")
    actual = generate_fixed_context(model, **values, max_new_tokens=2, eos_token_id=[])
    assert actual.shape == (1, 5)
    assert model._context is None


@pytest.mark.parametrize("failure", [None, "base", "writer"])
def test_cleanup_modes_weights_and_existing_gradients_survive_all_exits(monkeypatch, failure):
    model = tiny("full_augmented")
    model.train()
    model.support_writer.eval()
    model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modes = [(module, module.training) for module in model.modules()]
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.full_like(first_parameter, 0.25)
    gradient = first_parameter.grad.clone()

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate decoding failure")

    if failure == "base":
        monkeypatch.setattr(model.base, "forward", fail)
    elif failure == "writer":
        monkeypatch.setattr(model.support_writer, "forward", fail)
    if failure is None:
        generate_fixed_context(model, **inputs(), max_new_tokens=2, eos_token_id=[])
    else:
        with pytest.raises(RuntimeError, match="deliberate decoding failure"):
            generate_fixed_context(model, **inputs(), max_new_tokens=2, eos_token_id=[])
    assert model._context is None
    assert [(module, module.training) for module in model.modules()] == modes
    assert model.base.is_gradient_checkpointing
    assert torch.equal(first_parameter.grad, gradient)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_inflight_training_context_is_rejected_without_being_released():
    model = tiny("full_augmented")
    values = inputs()
    batch = {**values, "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
             "attention_mask": torch.ones(1, 5, dtype=torch.long),
             "labels": torch.tensor([[-100, -100, -100, 4, 5]])}
    output = model(**batch)
    context = model._context
    with pytest.raises(RuntimeError, match="still in flight"):
        generate_fixed_context(model, **values)
    assert model._context is context
    output.loss.backward()
    model.release_context()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("change", [
    {"max_new_tokens": 0}, {"max_new_tokens": True}, {"eos_token_id": True},
    {"eos_token_id": [48]}, {"prompt_ids": torch.tensor([[1, 2], [1, 2]])},
    {"prompt_mask": torch.tensor([[1, 1, 0]])},
    {"prompt_mask": torch.tensor([[1, 0, 1]])},
    {"prompt_mask": torch.tensor([[1., 1., 1.]])},
    {"memory_mask": torch.tensor([[1, 2]])},
    {"memory_ids": None, "memory_mask": torch.ones(1, 2, dtype=torch.bool)},
])
def test_invalid_request_fails_before_acquiring_context(change):
    model = tiny()
    values = {**inputs(), **change}
    with pytest.raises(ValueError):
        generate_fixed_context(model, **values)
    assert model._context is None
    assert model.training


def test_nonfinite_logits_raise_and_release_context(monkeypatch):
    model = tiny()

    def contaminate(module, args, output):
        output.logits[:, -1, 0] = float("nan")

    handle = model.base.register_forward_hook(contaminate)
    try:
        with pytest.raises(ValueError, match="nonfinite"):
            generate_fixed_context(model, **inputs())
    finally:
        handle.remove()
    assert model._context is None
