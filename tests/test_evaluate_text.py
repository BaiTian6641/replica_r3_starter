"""Strict fixture scoring must not find an answer inside explanatory text."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "evaluate_text", Path(__file__).parents[1] / "scripts/evaluate_text.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


@pytest.mark.parametrize("text", ["The answer is 2", "2 because XOR", "2\n3", "2.0",
                                 "<think></think>2", "```\n2\n```", "", "²", "٢"])
def test_integer_task_rejects_non_answer_text(text):
    assert evaluate.strict_integer(text) is None
    assert not evaluate.score_answer(text, "2")["task_correct"]


def test_integer_and_text_metrics_remain_distinct():
    assert evaluate.score_answer(" \n2\t", "2")["exact_text_match"]
    result = evaluate.score_answer("+02", "2")
    assert result["task_correct"] and result["strict_integer_match"]
    assert not result["exact_text_match"]
    assert not evaluate.score_answer("3", "2")["task_correct"]
    assert evaluate.score_answer(" done \n", "done")["task_correct"]
    assert evaluate.score_answer("done", "done")["strict_integer_match"] is None


def test_export_provenance_rejects_augmented_or_unrelated_parent(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"fixture")
    lock = {"repo_id": "source", "revision": "same", "config_sha256": "config"}
    manifest = {"regime": "full_augmented", "training_scope": "text_only",
                "run": {"model_lock": lock}}
    path = tmp_path / "adapter_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="D0 full"):
        evaluate.exported_inventory(base, {}, lock)
    manifest["regime"] = "full"
    manifest["run"]["model_lock"] = {**lock, "revision": "other"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="parent does not match"):
        evaluate.exported_inventory(base, {}, lock)


def test_split_parser_preserves_default_dev_and_accepts_explicit_train():
    required = ["--model-path", "source", "--model-lock", "lock.json",
                "--data", "data.jsonl", "--output", "output"]
    assert evaluate.build_parser().parse_args(required).split == "dev"
    assert evaluate.build_parser().parse_args(required + ["--split", "train"]).split == "train"
    with pytest.raises(SystemExit):
        evaluate.build_parser().parse_args(required + ["--split", "audit"])


def test_training_report_role_does_not_claim_development_accuracy():
    training = evaluate.evaluation_metadata("train", 8)
    assert training["evaluated_split"] == "train"
    assert training["evaluation_role"] == "tiny_training_memorization"
    assert training["available_examples"] == 8
    assert "available_dev_examples" not in training
    assert "not held-out accuracy" in training["limitations"][0]
    development = evaluate.evaluation_metadata("dev", 16)
    assert development["evaluation_role"] == "development_fixture_preflight"
    assert development["available_examples"] == development["available_dev_examples"] == 16
    with pytest.raises(ValueError, match="split"):
        evaluate.evaluation_metadata("audit", 8)


def test_explicit_training_split_keeps_loader_validation(tmp_path):
    row = {"id": "train-example", "group_id": "train-family", "split": "train",
           "prompt": "What changed?", "response": "81273", "verified": True,
           "memory_texts": ["Observed source record."], "source_ids": ["fixture://train"]}
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps(row) + "\n")
    assert evaluate.load_rows(data, "train") == [row]
    with pytest.raises(ValueError, match="split=dev"):
        evaluate.load_rows(data, "dev")

    class Tokenizer:
        eos_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            content = messages[0]["content"]
            assert "81273" not in content
            return content

        def encode(self, text, **kwargs):
            return [9] if text == "81273" else [2, 3]

    encoded = evaluate.encode_row(row, Tokenizer(), 32)
    assert 9 not in encoded["prompt_ids"][0].tolist()
    assert encoded["input_ids"][0].tolist()[-2:] == [9, 1]
    assert encoded["labels"][0].tolist()[-2:] == [9, 1]
