"""Exercise the native qualification workflow with a real tiny CPU Llama."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generation_qualification", ROOT / "scripts/qualify_augmented_generation.py")
QUALIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFY)


class Tokenizer:
    eos_token_id = 1

    def get_chat_template(self):
        return "tiny-known-user-prompt-with-assistant-prefix"

    def apply_chat_template(self, messages, **kwargs):
        assert "987654" not in messages[0]["content"]
        return "rendered observed prompt"

    def encode(self, text, **kwargs):
        if text == "987654":
            return [9]
        if text == "rendered observed prompt":
            return [0, 3, 4, 5]
        return [7, 8]

    def decode(self, ids, **kwargs):
        return " ".join(str(value) for value in ids)

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text('{"version":"1.0"}')
        (directory / "tokenizer_config.json").write_text('{"eos_token":"</s>"}')


class Profiler:
    def __init__(self):
        self.phases = []

    def mark(self, phase, **fields):
        self.phases.append(phase)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(QUALIFY.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(29)
    config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=64,
                         attention_dropout=0, bos_token_id=0, eos_token_id=1,
                         pad_token_id=2, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    native = LlamaForCausalLM(config)
    count = sum(value.numel() for value in native.parameters())
    source = tmp_path / "source"
    native.save_pretrained(source, safe_serialization=True)
    Tokenizer().save_pretrained(source)
    lock = {"repo_id": "test/native-tiny", "revision": "a" * 40,
            "config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
            "snapshot_files_sha256": {"config.json": hashlib.sha256((source / "config.json").read_bytes()).hexdigest()}}
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))

    def validate(path, claimed):
        assert Path(path) == source and claimed == lock
        assert hashlib.sha256((source / "config.json").read_bytes()).hexdigest() == claimed["config_sha256"]
    monkeypatch.setattr(QUALIFY, "validate_model_lock", validate)
    data = tmp_path / "dev.jsonl"
    data.write_text(json.dumps({"id": "dev-case", "group_id": "dev-source", "split": "dev",
                               "prompt": "Use the observed source.", "response": "987654",
                               "memory_texts": ["An observed rule."], "source_ids": ["fixture://dev"],
                               "verified": True}) + "\n")
    output = tmp_path / "output"
    output.mkdir()
    args = SimpleNamespace(model_path=source, model_lock=lock_path, data=data, split="dev",
                           output=output, device="cpu", max_length=32, max_new_tokens=4,
                           seed=17, rtol=0.0, atol=0.0)
    yield args, count
    torch.set_num_threads(threads)


def test_real_tiny_native_qualification_parity_context_sensitivity_and_reload(setup):
    args, count = setup
    phases = Profiler()
    report = QUALIFY.execute(args, phases, expected_original_parameters=count)
    assert report["status"] == "passed_tiny_augmented_generation_qualification"
    assert report["optimizer_updates"] == 0 and report["decoding"]["use_cache"] is False
    assert report["source_seal_validated"] and report["native_inventory"]["parameters"] == count
    assert report["example"]["prompt_tokens"] == 4
    assert report["example"]["sequence_tokens"] == 6
    assert [arm["regime"] for arm in report["arms"]] == ["modules_only", "full_augmented"]
    for arm in report["arms"]:
        assert arm["native_uncached_parity"]["generated_tokens_exact"]
        assert arm["native_uncached_parity"]["maximum_absolute_logit_difference"] == 0
        assert arm["forced_multitoken_parity"]["compared_decode_steps"] == 3
        assert arm["context_audit"]["context_build_calls"] == {"support_writer": 1, "workspace_encoder": 1}
        assert arm["context_audit"]["inputs_unchanged"]
        assert arm["context_audit"]["context_released"]
    assert report["arms"][0]["scope"]["original"]["trainable"] == 0
    assert report["arms"][1]["scope"]["original"]["trainable"] == count
    for branch in ("memory", "workspace"):
        assert report["nonzero_sidecar_sensitivity"][branch]["maximum_first_token_logit_difference"] > 0
    assert report["export_reload"]["maximum_absolute_logit_difference"] == 0
    assert report["export_reload"]["nonzero_sidecars"]
    assert "sidecars.safetensors" in report["export_reload"]["files_sha256"]
    assert phases.phases.index("original_backbone_released") < phases.phases.index("export_reload_generation_qualified")


def test_default_native_inventory_guard_cannot_be_bypassed_by_cli(setup):
    args, _ = setup
    with pytest.raises(ValueError, match="Unexpected native MiniCPM5 inventory"):
        QUALIFY.execute(args, Profiler())
    flags = ["--model-path", "source", "--model-lock", "lock", "--output", "output",
             "--expected-original-parameters", "123"]
    with pytest.raises(SystemExit):
        QUALIFY.build_parser().parse_args(flags)


def test_logit_or_token_mismatch_fails_qualification():
    reference = {"sequence": torch.tensor([[0, 3, 4]]), "step_logits": [torch.tensor([[1., 2.]])]}
    with pytest.raises(AssertionError, match="Generated tokens differ"):
        QUALIFY.assert_generation_parity({**reference, "sequence": torch.tensor([[0, 3, 5]])}, reference)
    with pytest.raises(AssertionError):
        QUALIFY.assert_generation_parity({**reference, "step_logits": [torch.tensor([[1., 3.]])]}, reference)
