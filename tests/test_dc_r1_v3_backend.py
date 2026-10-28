"""DC-R1 v3 native-normalization and raw-EOS backend regressions."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from dc_protocol_common_20261027 import read_json
from replica_lab import dc_r1_v3_scoring as scoring
from dc_protocol_common_20261027 import tokenizer_binding_sha
from tokenizer_fixtures import tiny_tokenizer
from replica_lab.dc_r1_v3_backend import NativeR1Backend

CONFIG = ROOT / "configs/dc_r1_native_value_protocol_v3.json"


class TinyAdapter:
    def __init__(self, base):
        self.base = base

    def parameter_stamp(self, original_only=False):
        return tuple((name, parameter.data_ptr(), parameter._version)
                     for name, parameter in self.base.named_parameters())


def _model() -> LlamaForCausalLM:
    torch.manual_seed(17)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=8, intermediate_size=16,
        num_hidden_layers=42, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0))
    model.model.norm.weight.data.copy_(torch.linspace(0.2, 1.7, 8))
    return model


def _backend(model, tmp_path):
    context = SimpleNamespace(config=read_json(CONFIG), root=ROOT, output=tmp_path)
    return NativeR1Backend(
        context, "dev", adapter=TinyAdapter(model), tokenizer=tiny_tokenizer())


class SequenceHead(torch.nn.Module):
    def __init__(self, sequence, vocab_size=128):
        super().__init__()
        self.sequence = list(sequence)
        self.vocab_size = vocab_size
        self.index = 0

    def forward(self, hidden):
        item = self.sequence[self.index]
        self.index += 1
        if isinstance(item, BaseException):
            raise item
        logits = hidden.new_zeros((1, 1, self.vocab_size))
        logits[0, -1, item] = 1.0
        return logits


def test_final_layer_logits_match_native_normal_path_with_nonuniform_norm(tmp_path):
    model = _model()
    backend = _backend(model, tmp_path)
    try:
        ids = [2, 3, 4]
        with torch.no_grad():
            actual = backend.next_logits(ids, 41)
            output = model.model(
                input_ids=torch.tensor([ids]), use_cache=False,
                output_hidden_states=True)
            expected = model.lm_head(
                model.model.norm(output.hidden_states[-1][:, -1:, :]))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert actual[0, -1].argmax().item() == expected[0, -1].argmax().item()
        assert backend.evaluate({"prompt": "x"}, layer=41,
                                prepared={"prompt_ids": ids})["normalization"] == (
                                    "native_model_norm")
    finally:
        backend.close()


def test_immediate_eos_is_empty_scored_output_with_charged_forward(tmp_path):
    model = _model()
    model.lm_head = SequenceHead([1])
    backend = _backend(model, tmp_path)
    steps = []
    try:
        result = backend.evaluate(
            {"prompt": "x"}, layer=13, prepared={"prompt_ids": [2, 3]},
            on_step=steps.append)
        assert result["raw_generated_ids"] == [1]
        assert result["terminal_stop_ids"] == [1]
        assert result["generated_ids"] == []
        assert result["generated_text"] == ""
        assert result["empty_generation"] is True
        assert result["stopped"] is True
        assert result["stop_reason"] == "eos"
        assert result["forward_calls"] == 1
        assert result["decode_forwards"] == 0
        assert result["generated_tokens"] == 1
        assert result["processed_positions"] == 2
        assert len(steps) == 1 and steps[0]["terminal"] is True
    finally:
        backend.close()


def test_content_plus_eos_preserves_raw_stop_and_content_cost(tmp_path):
    model = _model()
    model.lm_head = SequenceHead([65, 1])
    backend = _backend(model, tmp_path)
    try:
        result = backend.evaluate(
            {"prompt": "x"}, layer=27, prepared={"prompt_ids": [2, 3]})
        assert result["raw_generated_ids"] == [65, 1]
        assert result["terminal_stop_ids"] == [1]
        assert result["generated_ids"] == [65]
        assert result["generated_text"] == "A"
        assert result["forward_calls"] == 2
        assert result["decode_forwards"] == 1
        assert result["generated_tokens"] == 2
        assert result["processed_positions"] == 5
    finally:
        backend.close()


def test_length_termination_has_no_terminal_id_and_full_cost(tmp_path):
    model = _model()
    model.lm_head = SequenceHead([65, 66, 67])
    backend = _backend(model, tmp_path)
    backend.max_new = 3
    try:
        result = backend.evaluate(
            {"prompt": "x"}, layer=41, prepared={"prompt_ids": [2, 3]})
        assert result["raw_generated_ids"] == [65, 66, 67]
        assert result["terminal_stop_ids"] == []
        assert result["generated_ids"] == [65, 66, 67]
        assert result["stopped"] is False
        assert result["stop_reason"] == "max_new_tokens"
        assert result["forward_calls"] == 3
        assert result["decode_forwards"] == 2
        assert result["processed_positions"] == 9
    finally:
        backend.close()


def test_interruption_retains_every_completed_forward_before_error(tmp_path):
    model = _model()
    model.lm_head = SequenceHead([65, RuntimeError("injected forward failure")])
    backend = _backend(model, tmp_path)
    steps = []
    errors = []
    try:
        with pytest.raises(RuntimeError, match="injected forward failure"):
            backend.evaluate(
                {"prompt": "x"}, layer=13, prepared={"prompt_ids": [2, 3]},
                on_step=steps.append, on_error=errors.append)
        assert len(steps) == 1
        assert steps[0]["token_id"] == 65
        assert steps[0]["processed_positions"] == 2
        assert steps[0]["forward_calls"] == 1
        assert len(errors) == 1
        assert errors[0]["processed_positions"] == 3
        assert errors[0]["forward_calls"] == 1
    finally:
        backend.close()


def test_length_termination_is_scored_as_incomplete_not_correct(tmp_path):
    model = _model()
    model.lm_head = SequenceHead([65, 66, 67])
    backend = _backend(model, tmp_path)
    backend.max_new = 3
    try:
        result = backend.evaluate(
            {"prompt": "x"}, layer=41, prepared={"prompt_ids": [2, 3]})
        from replica_lab.dc_r1_v3_runner import _work_from_result
        result["work"] = _work_from_result(result)
        result["physical_stream_id"] = "exact_context/length-case/layer:41"
        config = read_json(CONFIG)
        binding = config["tokenizer_binding"]
        evidence = scoring._layer_evidence(
            result, condition="exact_context", case_id="length-case",
            target="A", binding=binding, decode_fn=backend.decode_fn,
            decode_metadata={
                "tokenizer_binding_sha256": tokenizer_binding_sha(binding),
                "decode_policy": binding["decode_policy"],
            })
        assert evidence["completion_valid"] is False
        assert evidence["correct"] is False
        assert evidence["work"]["forward_calls"] == 3
        assert evidence["work"]["processed_positions"] == 9
    finally:
        backend.close()
