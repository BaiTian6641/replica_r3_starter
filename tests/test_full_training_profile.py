"""Check scope ownership and batch/loss preservation for the profiling harness."""
import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location("profile_full_training", Path(__file__).parents[1] / "scripts/profile_full_training.py")
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def test_full_scope_rejects_freezing_missing_and_duplicate_optimizer_ownership():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    PROFILE.assert_full_scope(model, optimizer, 8, 2)
    model.bias.requires_grad_(False)
    with pytest.raises(ValueError, match="scope"):
        PROFILE.assert_full_scope(model, optimizer, 8, 2)
    model.bias.requires_grad_(True)
    optimizer.param_groups[0]["params"] = [model.weight]
    with pytest.raises(ValueError, match="own"):
        PROFILE.assert_full_scope(model, optimizer, 8, 2)
    optimizer.param_groups[0]["params"] = [model.weight, model.weight, model.bias]
    with pytest.raises(ValueError, match="own"):
        PROFILE.assert_full_scope(model, optimizer, 8, 2)


def test_microbatch_collation_preserves_causal_masks_and_response_labels():
    examples = [
        {"input_ids": torch.tensor([[4, 5, 6]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
         "labels": torch.tensor([[-100, 5, 6]])},
        {"input_ids": torch.tensor([[7, 8]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
         "labels": torch.tensor([[-100, 8]])},
    ]
    combined = PROFILE.collate_full_batches(examples, 1)
    assert combined["input_ids"].tolist() == [[4, 5, 6], [7, 8, 1]]
    assert combined["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert combined["labels"].tolist() == [[-100, 5, 6], [-100, 8, -100]]
    assert int((combined["labels"][:, 1:] != -100).sum()) == 3
