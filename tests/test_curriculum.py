"""Balanced sampling must remove update-level label skew without input leaks."""
import copy
import json
from pathlib import Path
import random

import pytest

from replica_lab.curriculum import (
    balanced_batches, batch_diagnostics, dataset_diagnostics, verify_xor_fixture,
)


ROOT = Path(__file__).resolve().parents[1]


def fixture_rows(split="train"):
    return [json.loads(line) for line in
            (ROOT / "data/fixtures" / f"{split}.jsonl").read_text().splitlines()]


def test_balanced_every_update_despite_unequal_dataset():
    rows = fixture_rows()
    diagnostics = dataset_diagnostics(rows)
    assert diagnostics["answer_classes"] == 16
    assert diagnostics["answer_histogram"]["0"] == 3
    assert diagnostics["answer_histogram"]["15"] == 1
    assert not diagnostics["answer_counts_equal"]
    for indices in balanced_batches(rows, grad_accum=16, steps=5, seed=17):
        actual = batch_diagnostics(rows, indices)
        assert actual["all_answer_classes_present"]
        assert actual["answer_counts_equal"]
        assert set(actual["answer_histogram"].values()) == {1}
        assert actual["example_ids"] == [rows[index]["id"] for index in indices]


def test_eight_row_overfit_subset_has_all_eight_answers_each_update():
    rows = [row for row in fixture_rows() if row["group_id"] == "xor-width-4-mask-1"]
    assert len(rows) == 8
    for indices in balanced_batches(rows, grad_accum=8, steps=3):
        assert sorted(indices) == list(range(8))
        assert batch_diagnostics(rows, indices)["answer_histogram"] == {
            str(answer): 1 for answer in range(8)}


def test_per_class_no_replacement_until_exhaustion():
    rows = fixture_rows()
    batches = list(balanced_batches(rows, 16, 3, seed=2))
    for answer in map(str, range(8)):
        selected = [index for indices in batches for index in indices
                    if rows[index]["response"] == answer]
        assert len(set(selected)) == 3


def test_sampling_is_deterministic_and_never_mutates_inputs_or_global_rng():
    rows = fixture_rows()
    original = copy.deepcopy(rows)
    state = random.getstate()
    first = list(balanced_batches(rows, 32, 4, seed=17))
    assert first == list(balanced_batches(rows, 32, 4, seed=17))
    assert first != list(balanced_batches(rows, 32, 4, seed=18))
    assert rows == original
    assert random.getstate() == state


@pytest.mark.parametrize("grad_accum", [0, 1, 8, 17, True, 16.0])
def test_rejects_accumulation_that_cannot_balance_all_classes(grad_accum):
    with pytest.raises(ValueError, match="grad_accum"):
        list(balanced_batches(fixture_rows(), grad_accum, 1))


def test_zero_steps_is_empty_but_parameters_are_still_validated():
    assert list(balanced_batches(fixture_rows(), 16, 0)) == []
    with pytest.raises(ValueError, match="steps"):
        list(balanced_batches(fixture_rows(), 16, -1))


def test_rejects_dev_training_and_explicit_source_leak():
    with pytest.raises(ValueError, match="split=train"):
        list(balanced_batches(fixture_rows("dev"), 8, 1))
    train, dev = fixture_rows()[0], fixture_rows("dev")[0]
    dev["source_ids"] = train["source_ids"]
    with pytest.raises(ValueError, match="Source crosses splits"):
        dataset_diagnostics([train, dev])


def test_all_shipped_fixture_answers_are_independently_verified():
    for row in fixture_rows() + fixture_rows("dev"):
        verified = verify_xor_fixture(row)
        assert verified["verified"] is True
        assert verified["expected"] == row["response"]


@pytest.mark.parametrize("field,value", [
    ("response", "wrong"),
    ("memory_texts", ["The gold answer is 1"]),
    ("prompt", "Input is 0. Gold answer is 1."),
    ("source_ids", ["fixture://unrelated"]),
    ("id", "different"),
])
def test_fixture_verifier_rejects_wrong_labels_and_input_or_provenance_edits(field, value):
    row = fixture_rows()[0]
    row[field] = value
    with pytest.raises(ValueError):
        verify_xor_fixture(row)


@pytest.mark.parametrize("indices", [[], [-1], [32], [True]])
def test_batch_diagnostics_rejects_invalid_indices(indices):
    with pytest.raises(ValueError):
        batch_diagnostics(fixture_rows(), indices)
