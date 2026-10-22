"""CPU-only tests of bounded native-checkpoint comparison, without model loads."""
import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location(
    "audit_checkpoint", Path(__file__).parents[1] / "scripts/audit_checkpoint.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_pair(tmp_path, source, exported):
    left, right = tmp_path / "source", tmp_path / "exported"
    left.mkdir()
    right.mkdir()
    save_file(source, str(left / "model.safetensors"))
    save_file(exported, str(right / "model.safetensors"))
    return left, right


def test_dtype_conversion_is_not_a_training_update(tmp_path):
    # Values are deliberately not all exactly representable in BF16 initially.
    source = {"model.embed_tokens.weight": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.bfloat16),
              "lm_head.weight": torch.tensor([1.5, -0.7], dtype=torch.bfloat16),
              "scalar": torch.tensor(0.3, dtype=torch.bfloat16)}
    left, right = write_pair(tmp_path, source, {name: value.float() for name, value in source.items()})
    result = audit.compare_checkpoints(left, right, expected_tensors=3,
                                       expected_parameters=7, chunk_bytes=8)
    assert result["changed_tensor_count"] == 0
    assert result["changed_elements"] == 0
    assert not result["every_original_tensor_changed"]
    assert not result["input_embedding_changed"]
    assert not result["output_head_changed"]
    assert all(item["max_abs_delta"] == 0 for item in result["tensors"].values())


def test_actual_changes_and_wide_tensor_chunking(tmp_path):
    source = {"model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).bfloat16(),
              "lm_head.weight": torch.ones(3, dtype=torch.bfloat16)}
    exported = {name: value.float() for name, value in source.items()}
    exported["model.embed_tokens.weight"][1, 2, 3] += 0.5
    exported["lm_head.weight"][0] -= 0.25
    left, right = write_pair(tmp_path, source, exported)
    result = audit.compare_checkpoints(left, right, chunk_bytes=8)
    assert result["changed_tensor_count"] == 2
    assert result["changed_elements"] == 2
    assert result["every_original_tensor_changed"]
    assert result["input_embedding_changed"] and result["output_head_changed"]
    assert result["tensors"]["model.embed_tokens.weight"]["max_abs_delta"] == 0.5
    assert result["tensors"]["lm_head.weight"]["max_abs_delta"] == 0.25
    assert set(result["exported_checkpoint_sha256"]) == {"model.safetensors"}


def test_name_mismatch_rejected(tmp_path):
    left, right = write_pair(tmp_path, {"left": torch.ones(2, dtype=torch.bfloat16)},
                              {"right": torch.ones(2)})
    with pytest.raises(ValueError, match="Tensor name mismatch"):
        audit.compare_checkpoints(left, right)


def test_shape_mismatch_rejected(tmp_path):
    left, right = write_pair(tmp_path, {"weight": torch.ones(2, dtype=torch.bfloat16)},
                              {"weight": torch.ones(3)})
    with pytest.raises(ValueError, match="Tensor shape mismatch"):
        audit.compare_checkpoints(left, right)
