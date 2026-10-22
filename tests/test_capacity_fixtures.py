"""Capacity data must have real native lengths, correct labels and no split leak."""
import importlib.util
import json
from pathlib import Path

import pytest

from replica_lab.contracts import audit_splits
from replica_lab.curriculum import dataset_diagnostics


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("capacity_builder", ROOT / "scripts/make_capacity_fixtures.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class WordTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return "user " + messages[0]["content"] + " assistant"

    def encode(self, text, **kwargs):
        return list(range(2, len(text.split()) + 2))


def test_all_episode_inputs_are_solver_checked_and_source_disjoint():
    rows = [BUILDER.make_episode(256, split, case, 8) for split in ("train", "dev") for case in range(8)]
    audit = audit_splits(rows)
    assert audit["split_counts"]["train"] == audit["split_counts"]["dev"] == 8
    assert not ({r["group_id"] for r in rows[:8]} & {r["group_id"] for r in rows[8:]})
    assert not ({r["source_ids"][0] for r in rows[:8]} & {r["source_ids"][0] for r in rows[8:]})
    for split in ("train", "dev"):
        selected = [row for row in rows if row["split"] == split]
        assert dataset_diagnostics(selected)["answer_histogram"] == {str(value): 1 for value in range(8)}
        for row in selected:
            checked = BUILDER.verify_episode(row)
            assert checked["expected"] == row["response"]
            assert checked["observed_write_count"] == 10
            assert "answer" not in " ".join(row["memory_texts"]).lower()
            assert "expected" not in " ".join(row["memory_texts"]).lower()


@pytest.mark.parametrize("mutation", ["label", "clock", "source", "evidence", "annotation"])
def test_solver_and_provenance_fail_closed_on_episode_edits(mutation):
    row = BUILDER.make_episode(256, "train", 0, 8)
    if mutation == "label":
        row["response"] = "7"
    elif mutation == "clock":
        row["memory_texts"][1] = row["memory_texts"][1].replace("Clock 0", "Clock 9")
    elif mutation == "source":
        row["source_ids"] = ["fixture://other"]
    elif mutation == "evidence":
        row["memory_texts"].pop(1)
    else:
        row["memory_texts"].append("Gold answer: 1.")
    with pytest.raises(ValueError):
        BUILDER.verify_episode(row)


def test_exact_length_search_preserves_complete_facts_and_rejects_impossible_budget():
    row, audit = BUILDER.fit_episode(WordTokenizer(), 120, "train", 0)
    assert audit["actual_sequence_tokens"] == 120
    assert audit["prompt_tokens"] + audit["supervised_tokens"] == 120
    assert audit["supervised_tokens"] == 2
    assert BUILDER.verify_episode(row)["distractor_writes"] >= 4
    with pytest.raises(ValueError, match="Cannot construct exact"):
        BUILDER.fit_episode(WordTokenizer(), 8, "train", 0)


def test_per_bit_solver_rejects_invalid_bus_bounds():
    for args in [(0, 0, 0), (33, 0, 0), (4, 16, 0), (4, 1, -1)]:
        with pytest.raises(ValueError):
            BUILDER.bitwise_truth(*args)
    for value in range(8):
        assert BUILDER.bitwise_truth(4, 1, value) == (value ^ 1)


def test_label_construction_and_per_bit_verifier_are_independent(monkeypatch):
    row = BUILDER.make_episode(256, "train", 0, 8)
    monkeypatch.setattr(BUILDER, "bitwise_truth", lambda *args: 7)
    assert BUILDER.make_episode(256, "train", 0, 8)["response"] == "1"
    with pytest.raises(ValueError, match="independent per-bit solver"):
        BUILDER.verify_episode(row)


def test_tokenizer_provenance_rejects_unsealed_and_changed_bytes(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"revision": "a" * 40, "local_bytes_sealed": False}))
    with pytest.raises(ValueError, match="sealed"):
        BUILDER.tokenizer_provenance(tmp_path, lock)
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "special_tokens_map.json"]
    for name in names:
        (tmp_path / name).write_text("original")
    payload = {"revision": "a" * 40, "repo_id": "test/tiny", "local_bytes_sealed": True,
               "snapshot_files_sha256": {name: BUILDER.digest(tmp_path / name) for name in names}}
    lock.write_text(json.dumps(payload))
    assert BUILDER.tokenizer_provenance(tmp_path, lock)["source_revision"] == "a" * 40
    (tmp_path / "chat_template.jinja").write_text("changed")
    with pytest.raises(ValueError, match="seal mismatch"):
        BUILDER.tokenizer_provenance(tmp_path, lock)


@pytest.mark.skipif(not (ROOT / "checkpoints/minicpm5_source/tokenizer.json").exists(),
                    reason="Native tokenizer is an ignored local acquisition")
@pytest.mark.parametrize("length", [256, 512])
def test_pinned_native_tokenizer_reaches_exact_capacity_lengths(length):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "checkpoints/minicpm5_source",
                                               local_files_only=True, trust_remote_code=False)
    rows, manifest = BUILDER.build_workload(tokenizer, length)
    assert manifest["quality_training"] is False
    assert manifest["actual_sequence_token_histogram"] == {length: 16}
    assert manifest["supervised_token_histogram"] == {2: 16}
    assert manifest["length_policy"]["padding_tokens"] == manifest["length_policy"]["truncated_tokens"] == 0
    for split in rows:
        for row in rows[split]:
            batch = BUILDER.encode_row(row, tokenizer, length)
            assert batch["input_ids"].shape[1] == length
            assert batch["prompt_ids"].shape[1] == length - 2
            assert batch["labels"][0, :-2].eq(-100).all()
