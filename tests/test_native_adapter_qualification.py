"""CPU-only checks of the native-wrapper qualification protocol, no downloads."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from replica_lab.model_adapter import LlamaMemoryAdapter


SCRIPT = Path(__file__).parents[1] / "scripts/qualify_native_adapter.py"
SPEC = importlib.util.spec_from_file_location("qualify_native_adapter", SCRIPT)
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def base_model():
    torch.manual_seed(83)
    config = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=64,
                         attention_dropout=0.0, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config)


def batch():
    return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "labels": torch.tensor([[-100, -100, -100, 4, 5]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "prompt_ids": torch.tensor([[1, 2, 3]]),
            "memory_ids": torch.tensor([[7, 8]]),
            "prompt_mask": torch.ones(1, 3, dtype=torch.bool),
            "memory_mask": torch.ones(1, 2, dtype=torch.bool)}


def adapter(base, regime):
    return LlamaMemoryAdapter(base, regime, memory_size=16, workspace_size=16,
                               slots=2, workspace_steps=2)


@pytest.mark.parametrize("regime", ["modules_only", "full_augmented"])
def test_real_tiny_native_null_parity_scope_causality_and_recompute(regime):
    base = base_model()
    data = batch()
    reference = qualification.native_logits(base, data)
    model = adapter(base, regime)
    original = {name: value.detach().clone()
                for name, value in model._original_parameters.items()}
    report = qualification.qualify_adapter(model, data, reference)
    assert report["null_parity"]["maximum_absolute_difference"] == 0
    assert report["nonzero_branch_sensitivity"]["maximum_difference"] > 0
    assert report["causality"] == {
        "target_suffix_context_invariance": "passed", "target_in_workspace_rejected": True,
        "source_features_rebuilt_after_writer_change": "passed", "context_released": True}
    assert report["checkpoint_backward"]["status"] == "passed"
    assert report["checkpoint_backward"]["forward_calls"]["decoder"] >= 2
    expected_original = report["scope"]["original"]["total"] if regime == "full_augmented" else 0
    assert report["scope"]["original"]["trainable"] == expected_original
    assert model._context is None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.count_nonzero(model.workspace.reader.output_projection.weight) == 0
    for name, parameter in model._original_parameters.items():
        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)


def test_parity_check_detects_nonzero_branch_instead_of_comparing_wrapper_to_itself():
    base = base_model()
    data = batch()
    native = qualification.native_logits(base, data)
    model = adapter(base, "modules_only")
    qualification._probe_projection(model)
    with pytest.raises(AssertionError):
        qualification.assert_null_parity(model, data, native)
    assert model._context is None
    with pytest.raises(AssertionError, match="exactly zero"):
        qualification.qualify_adapter(model, data, native, backward=False)


def test_nonzero_sidecar_roundtrip_after_releasing_original(tmp_path):
    model = adapter(base_model(), "full_augmented")
    qualification._probe_projection(model)
    expected, _ = qualification.adapter_forward(model, batch())
    model.save_pretrained(tmp_path)
    del model
    loaded = adapter(LlamaForCausalLM.from_pretrained(
        tmp_path / "base", local_files_only=True, attn_implementation="eager"), "full_augmented")
    loaded.load_sidecars(tmp_path / "sidecars.safetensors")
    actual, _ = qualification.adapter_forward(loaded, batch())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bool(torch.count_nonzero(loaded.workspace.reader.output_projection.weight))
    assert loaded._context is None


@pytest.mark.parametrize("mutation", ["target_prompt", "missing_labels", "wrong_prefix"])
def test_input_boundary_rejects_future_targets(mutation):
    data = batch()
    if mutation == "target_prompt":
        data["prompt_ids"] = data["input_ids"]
        data["prompt_mask"] = torch.ones_like(data["input_ids"], dtype=torch.bool)
    elif mutation == "missing_labels":
        del data["labels"]
    else:
        data["prompt_ids"] = torch.tensor([[7, 8, 9]])
    with pytest.raises(ValueError):
        qualification.validate_causal_batch(data)


def test_native_cli_inventory_guard_rejects_tiny_substitution():
    with pytest.raises(ValueError, match="Unexpected native MiniCPM5 inventory"):
        qualification.check_native_inventory(base_model())


def test_failure_report_preserves_arguments_scope_and_source_data_receipts(tmp_path, monkeypatch):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"repo_id": "openbmb/MiniCPM5-2B", "revision": "a" * 40,
                               "snapshot_files_sha256": {"config.json": "claimed"}}))
    data = tmp_path / "data.jsonl"
    data.write_text('{"id": "failed-input"}\n')
    output = tmp_path / "qualification"

    class FakeProfiler:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            return self
        def stop(self):
            return {"device": "cpu"}

    def fail(args, profiler, **kwargs):
        raise ValueError("deliberate validation failure")

    monkeypatch.setattr(qualification, "RunProfiler", FakeProfiler)
    monkeypatch.setattr(qualification, "execute", fail)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--model-path", str(tmp_path),
        "--model-lock", str(lock), "--data", str(data), "--output", str(output), "--device", "cpu"])
    with pytest.raises(ValueError, match="deliberate validation failure"):
        qualification.main()
    report = json.loads((output / "qualification_report.json").read_text())
    assert report["status"] == "failed_native_adapter_qualification"
    assert report["device"] == "cpu"
    assert report["source_seal_validated"] is False
    assert report["arguments"]["data"] == str(data)
    assert "D1/D2" in report["qualification_scope"]
    assert report["source_lock_receipt"]["sha256"] == qualification.file_digest(lock)
    assert report["data_receipt"]["sha256"] == qualification.file_digest(data)
    assert report["claimed_source_lock"]["snapshot_files_sha256"] == {"config.json": "claimed"}
    assert report["implementation_sha256"]["scripts/qualify_native_adapter.py"] == qualification.file_digest(SCRIPT)
