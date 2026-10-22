"""Complete checkpoint receipts bind native weights and resumable run state."""
import copy
import json
from pathlib import Path
import random

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from replica_lab.checkpointing import (
    file_digest, save_training_checkpoint, validate_training_checkpoint,
)
from replica_lab.model_adapter import LlamaMemoryAdapter
from replica_lab.optimizers import build_optimizer
from replica_lab.training_state import load_runtime_state


class SmallTokenizer:
    """Only serialization is needed: this test never tokenizes text."""

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text('{"version":"1.0"}\n')
        (directory / "tokenizer_config.json").write_text('{"eos_token":"</s>"}\n')


@pytest.fixture(autouse=True)
def cpu_rng_only(monkeypatch):
    # Runtime capture follows all available backends. Restrict this CPU-only
    # test process so no accelerator state is initialized on the host Mac.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def tiny_model():
    torch.manual_seed(23)
    config = LlamaConfig(vocab_size=24, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=32,
                         attention_dropout=0.0, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    return LlamaMemoryAdapter(LlamaForCausalLM(config), "full")


def optimizer_for(model):
    return build_optimizer(model, lr_base=1e-4, lr_muon_base=1e-4,
                           weight_decay=0.01, block_size=64)


def update(model, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    model(input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
          attention_mask=torch.ones(1, 5, dtype=torch.long),
          labels=torch.tensor([[-100, -100, -100, 4, 5]])).loss.backward()
    model.release_context()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def contract():
    return {
        "model_source": {"repo_id": "test/native-llama", "revision": "sealed-revision",
                         "config_sha256": "a" * 64, "weights_sha256": "b" * 64},
        "tokenizer_sha256": "c" * 64,
        "train_sha256": "d" * 64,
        "dev_sha256": "e" * 64,
        "regime": "full", "optimizer": "muon8bit",
        "lr_base": 1e-4, "lr_muon_base": 1e-4,
        "grad_accum": 8, "seed": 17,
    }


def checkpoint(tmp_path):
    model = tiny_model()
    optimizer = optimizer_for(model)
    update(model, optimizer)
    destination = tmp_path / "step-0001"
    info = save_training_checkpoint(
        model, SmallTokenizer(), optimizer, destination, step=1,
        data_state={"completed_updates": 1, "batch_cursor": 8},
        schedule_state={"kind": "constant", "step": 1},
        run_contract=contract(), manifest={"purpose": "checkpoint test"})
    return destination, info, model, optimizer


def rewrite_receipt(root, mutate):
    path = root / "checkpoint_receipt.json"
    receipt = json.loads(path.read_text())
    mutate(receipt)
    path.write_text(json.dumps(receipt))


def test_complete_native_checkpoint_inventory_and_runtime_roundtrip(tmp_path):
    path, info, _, optimizer = checkpoint(tmp_path)
    receipt = validate_training_checkpoint(path, contract())
    assert info["step"] == receipt["completed_steps"] == 1
    assert info["receipt_sha256"] == file_digest(path / "checkpoint_receipt.json")
    assert {"adapter_manifest.json", "base/config.json", "base/model.safetensors",
            "base/tokenizer.json", "base/tokenizer_config.json", "runtime.pt"}.issubset(
                receipt["files_sha256"])
    assert all(len(digest) == 64 for digest in receipt["files_sha256"].values())
    assert "checkpoint_receipt.json" not in receipt["files_sha256"]
    state = torch.load(path / "runtime.pt", weights_only=True, map_location="cpu")
    assert state["metadata"]["contract_sha256"] == receipt["contract_sha256"]
    assert state["metadata"]["data_state"]["batch_cursor"] == 8
    assert state["metadata"]["schedule_state"] == {"kind": "constant", "step": 1}
    assert optimizer.audit()["state_bytes_by_dtype"]["torch.int8"] > 0
    assert optimizer.audit()["state_bytes_by_dtype"]["torch.float32"] > 0


def test_resumed_native_weights_and_mixed_optimizer_match_next_update(tmp_path):
    path, _, uninterrupted, original_optimizer = checkpoint(tmp_path)
    expected_python = random.random()
    expected_torch = torch.rand(4)
    update(uninterrupted, original_optimizer)
    validate_training_checkpoint(path, contract())
    native = AutoModelForCausalLM.from_pretrained(
        path / "base", local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", use_safetensors=True)
    resumed = LlamaMemoryAdapter(native, "full")
    resumed_optimizer = optimizer_for(resumed)
    metadata = load_runtime_state(path / "runtime.pt", resumed_optimizer,
                                  run_contract=contract())
    assert metadata["step"] == 1
    assert metadata["data_state"]["completed_updates"] == 1
    assert random.random() == expected_python
    torch.testing.assert_close(torch.rand(4), expected_torch, atol=0, rtol=0)
    assert any(value.dtype == torch.int8
               for state in resumed_optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor))
    update(resumed, resumed_optimizer)
    for (name, expected), (actual_name, actual) in zip(
            uninterrupted.named_parameters(), resumed.named_parameters()):
        assert name == actual_name
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert resumed_optimizer.audit() == original_optimizer.audit()


def test_checkpoint_bytes_and_missing_or_extra_inventory_fail_closed(tmp_path):
    path, _, _, _ = checkpoint(tmp_path)
    weights = path / "base/model.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(original + b"corrupt")
    with pytest.raises(ValueError, match="byte mismatch"):
        validate_training_checkpoint(path, contract())
    weights.write_bytes(original)
    weights.unlink()
    with pytest.raises(ValueError, match="inventory"):
        validate_training_checkpoint(path, contract())
    weights.write_bytes(original)
    (path / "unexpected.txt").write_text("not in the receipt")
    with pytest.raises(ValueError, match="inventory"):
        validate_training_checkpoint(path, contract())


@pytest.mark.parametrize("field", ["model_source", "tokenizer_sha256", "train_sha256",
                                 "dev_sha256", "grad_accum", "lr_muon_base"])
def test_source_data_and_training_contract_changes_fail_closed(tmp_path, field):
    path, _, _, _ = checkpoint(tmp_path)
    changed = copy.deepcopy(contract())
    changed[field] = {"revision": "other"} if field == "model_source" else "other"
    with pytest.raises(ValueError, match="contract"):
        validate_training_checkpoint(path, changed)


def test_existing_checkpoint_is_never_overwritten(tmp_path):
    path, _, model, optimizer = checkpoint(tmp_path)
    before = file_digest(path / "checkpoint_receipt.json")
    with pytest.raises(FileExistsError):
        save_training_checkpoint(
            model, SmallTokenizer(), optimizer, path, step=2,
            data_state={}, schedule_state=None, run_contract=contract(), manifest={})
    assert file_digest(path / "checkpoint_receipt.json") == before


@pytest.mark.parametrize("missing", ["base/model.safetensors", "base/config.json",
                                    "base/tokenizer.json"])
def test_missing_native_artifacts_fail_even_if_receipt_inventory_is_rewritten(tmp_path, missing):
    path, _, _, _ = checkpoint(tmp_path)
    (path / missing).unlink()
    rewrite_receipt(path, lambda receipt: receipt["files_sha256"].pop(missing))
    with pytest.raises(ValueError, match="model|weight|native|config|tokenizer"):
        validate_training_checkpoint(path, contract())


def test_receipt_contract_digest_is_checked(tmp_path):
    path, _, _, _ = checkpoint(tmp_path)
    rewrite_receipt(path, lambda receipt: receipt.update(contract_sha256="0" * 64))
    with pytest.raises(ValueError, match="contract|digest|hash"):
        validate_training_checkpoint(path, contract())


def test_receipt_step_must_match_runtime_completed_step(tmp_path):
    path, _, _, _ = checkpoint(tmp_path)
    rewrite_receipt(path, lambda receipt: receipt.update(completed_steps=999))
    with pytest.raises(ValueError, match="step|runtime"):
        validate_training_checkpoint(path, contract())


def test_nested_receipt_named_file_is_part_of_inventory(tmp_path):
    path, _, _, _ = checkpoint(tmp_path)
    (path / "base/checkpoint_receipt.json").write_text("unexpected nested data")
    with pytest.raises(ValueError, match="inventory"):
        validate_training_checkpoint(path, contract())


def test_interrupted_save_never_publishes_resumable_destination(tmp_path):
    class InterruptedModel:
        def save_pretrained(self, directory, *, manifest):
            (Path(directory) / "partial.txt").write_text("incomplete")
            raise RuntimeError("simulated interrupted write")

    model = tiny_model()
    destination = tmp_path / "step-0002"
    with pytest.raises(RuntimeError, match="interrupted"):
        save_training_checkpoint(
            InterruptedModel(), SmallTokenizer(), optimizer_for(model), destination,
            step=2, data_state={}, schedule_state=None,
            run_contract=contract(), manifest={})
    assert not destination.exists()
    partials = list(tmp_path.glob(".step-0002-*"))
    assert len(partials) == 1
    assert (partials[0] / "partial.txt").exists()
    assert not (partials[0] / "checkpoint_receipt.json").exists()
    with pytest.raises((OSError, ValueError)):
        validate_training_checkpoint(partials[0], contract())
