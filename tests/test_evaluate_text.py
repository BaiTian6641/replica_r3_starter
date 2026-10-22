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
