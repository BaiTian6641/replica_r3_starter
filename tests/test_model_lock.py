"""Metadata-seal tests with dummy files; these are not model checkpoints."""
import hashlib
import json
import pytest
pytest.importorskip("torch")
from replica_lab.model_adapter import validate_model_lock


def fixture(tmp_path):
    config = {"model_type": "llama", "hidden_size": 2048, "num_hidden_layers": 42,
              "vocab_size": 130560, "tie_word_embeddings": False,
              "architectures": ["LlamaForCausalLM"]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"dummy metadata-only test, not valid weights")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    lock = {"repo_id": "openbmb/MiniCPM5-2B", "revision": "0"*40,
            "config_sha256": hashes["config.json"], "snapshot_files_sha256": hashes}
    return lock


def test_config_match_does_not_hide_changed_weights(tmp_path):
    lock = fixture(tmp_path)
    validate_model_lock(tmp_path, lock)
    (tmp_path / "model.safetensors").write_bytes(b"changed dummy bytes")
    with pytest.raises(ValueError, match="byte mismatch"):
        validate_model_lock(tmp_path, lock)


def test_unsealed_snapshot_fails_closed(tmp_path):
    lock = fixture(tmp_path)
    lock.pop("snapshot_files_sha256")
    with pytest.raises(ValueError, match="Seal local"):
        validate_model_lock(tmp_path, lock)


def test_added_tokenizer_file_invalidates_seal(tmp_path):
    lock = fixture(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory differs"):
        validate_model_lock(tmp_path, lock)
