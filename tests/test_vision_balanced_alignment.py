"""Deterministic CPU checks for the balanced alignment diagnostic."""
from __future__ import annotations

import copy
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import time

import pytest

import replica_lab.vision_balanced_alignment as vba
from replica_lab.vision_curriculum import (
    build_procedural, load_config as load_curriculum_config)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/vision_balanced_alignment_v1.json"


@lru_cache(maxsize=1)
def _config():
    return vba.load_config(CONFIG_PATH)


@lru_cache(maxsize=1)
def _built():
    return vba.build_balanced_dataset(_config(), ROOT)


@lru_cache(maxsize=1)
def _procedural():
    config = copy.deepcopy(load_curriculum_config(
        ROOT / "configs/vision_curriculum_v1.json"))
    overrides = _config()["procedural_overrides"]
    config["seed"] = _config()["seed"]
    config["procedural"]["id_prefix"] = overrides["id_prefix"]
    config["procedural"]["worlds_per_task"] = overrides["worlds_per_task"]
    config["procedural"]["renderer"] = {
        **config["procedural"]["renderer"],
        "width": overrides["renderer"]["width"],
        "height": overrides["renderer"]["height"],
    }
    config["procedural"]["split_layouts"] = {
        split: list(values) for split, values in
        overrides["split_layouts"].items()}
    return build_procedural(config)


def test_dataset_balance_freshness_and_labels():
    built = _built()
    counts = built["manifest"]["counts"]
    assert counts["train_answers"] == 28
    assert counts["train_captions"] == 18
    assert counts["train_examples"] == 46
    assert counts["train_updates"] == 10
    assert counts["train_examples_scheduled"] == 60
    assert counts["dev_answers"] == 27
    assert counts["dev_captions"] == 12
    assert counts["heldout_answers"] == 18
    assert counts["heldout_captions"] == 12
    selection = _config()["selection"]
    for role_name in ("train", "dev", "heldout"):
        role = built["balance"][role_name]
        assert set(role["answer_histogram_selected"].values()) \
            == {selection[f"{role_name}_answers_per_class"]}
        assert set(role["caption_histogram_selected"].values()) \
            == {selection[f"{role_name}_captions_per_class"]}
        assert all(key.startswith("objects=")
                   for key in role["caption_histogram_selected"])
    assert built["freshness"]["namespace_disjoint_from_prior"] is True
    for split in ("train", "dev", "heldout"):
        for metric in ("case_overlap", "image_overlap", "scene_overlap"):
            assert built["freshness"][split][metric] == 0
    assert built["freshness"]["all_used_cases"]["image_overlap"] == 0
    assert built["freshness"]["all_used_prior_assets"]["image_overlap"] == 0
    assert vba.question_leaks("Is the red circle present?", "red") is True
    assert vba.question_leaks("Is the red circle present?", "blue") is False
    assert all(not vba.question_leaks(case["question"], label["answer"])
               for case, label in zip(
                   built["heldout_answer_cases"],
                   built["heldout_answer_labels"]))
    assert built["freshness"]["all_used_prior_assets"]["geometry"] == {
        "width": 160, "height": 120}
    assert built["semantic_audit"]["all_passed"] is True
    assert built["semantic_audit"]["visible_pixels_verified"] > 0
    for case in (built["dev_answer_cases"] + built["heldout_answer_cases"]
                 + built["dev_caption_cases"] + built["heldout_caption_cases"]):
        assert "answer" not in case and "label" not in case
        assert "response" not in case
        assert case["case_id"].startswith("vba1-")
        assert len(case["image_sha256"]) == 64
        assert len(case["counterfactual_image_sha256"]) == 64
    for label in (built["dev_answer_labels"] + built["heldout_answer_labels"]
                  + built["dev_caption_labels"]
                  + built["heldout_caption_labels"]):
        assert label["label_visibility"] == "verifier_only"
    for label in built["heldout_caption_labels"]:
        assert label["caption_changes_under_counterfactual"] is True
        assert label["answer"] != label["counterfactual_answer"]
    for label in (built["dev_answer_labels"]
                  + built["heldout_answer_labels"]):
        if label["task"] != "abstain":
            assert label["answer_changes_under_counterfactual"] is True
            assert label["answer"] != label["counterfactual_answer"]
    for row in built["train_batches"]:
        assert len(row["example_ids"]) == 6
        assert len(set(row["example_ids"])) == 6
    scheduled = [example_id for batch in built["train_batches"]
                 for example_id in batch["example_ids"]]
    assert set(scheduled) == {row["example_id"]
                              for row in built["train_answer_rows"]
                              + built["train_caption_rows"]}
    assert built["manifest"]["files"][
        "training/train_caption_labels.jsonl"]["bytes"] > 0


def test_caption_semantics_and_pixel_rejection():
    procedural = _procedural()
    assets = {row["asset_id"]: row for row in procedural["assets"]}
    scenes = {row["asset_id"]: row["scene"] for row in procedural["scenes"]}
    scene_entry = next(row for row in procedural["scenes"]
                       if any(obj.get("role") == "occluded"
                              for obj in row["scene"]["objects"]))
    scene = scene_entry["scene"]
    asset_id = scene_entry["asset_id"]
    caption = vba.scene_caption(scene)
    check = vba.verify_caption_semantics(caption, scene)
    assert check["exact_match"] is True
    assert all(descriptor in [list(item) for item in
                              vba.visible_descriptors(scene)]
               for descriptor in check["parsed_descriptors"])
    occluded = vba.occluded_descriptors(scene)
    assert occluded
    bad = caption + f" and a {occluded[0][0]} {occluded[0][1]}"
    with pytest.raises(ValueError):
        vba.verify_caption_semantics(bad, scene)
    with pytest.raises(ValueError):
        vba.verify_caption_semantics("a red circle", scene)
    png = procedural["files"][assets[asset_id]["path"]]
    pixels = vba.verify_scene_pixels(png, scene)
    assert pixels["pixel_check_passed"] is True
    assert pixels["occluded_pixels_inspected"] == len(occluded)
    assert pixels["visible_pixels_verified"] == len(
        vba.visible_descriptors(scene))
    bad_scene = copy.deepcopy(scene)
    visible = next(obj for obj in bad_scene["objects"]
                   if obj.get("role") != "occluded")
    visible["color"] = "red" if visible["color"] != "red" else "blue"
    with pytest.raises(ValueError):
        vba.verify_scene_pixels(png, bad_scene)
    agreement = vba.semantic_caption_agreement(
        caption, [list(item) for item in vba.visible_descriptors(scene)])
    assert agreement["exact_agreement"] is True
    assert agreement["f1"] == 1.0
    phantom = vba.semantic_caption_agreement(
        caption + " and a orange triangle",
        [list(item) for item in vba.visible_descriptors(scene)])
    assert phantom["phantom_objects"] == 1
    assert phantom["exact_agreement"] is False


def test_balanced_selection_rejects_short_classes():
    rows = [{"case_id": "a", "cls": 1}, {"case_id": "b", "cls": 1},
            {"case_id": "c", "cls": 2}]
    selected, histogram = vba.balanced_select(
        rows, lambda row: row["cls"], per_class=1)
    assert [row["case_id"] for row in selected] == ["a", "c"]
    assert histogram == {"1": 2, "2": 1}
    with pytest.raises(ValueError):
        vba.balanced_select(rows, lambda row: row["cls"], per_class=2)


def _caption_label(case_id, world_id, caption, objects):
    return {"case_id": case_id, "world_id": world_id, "answer": caption,
            "visible_objects": objects, "label_visibility": "verifier_only"}


def test_caption_control_scoring_and_semantic_margins():
    labels = [
        _caption_label("a", "w1", "a red circle",
                       [["red", "circle"]]),
        _caption_label("b", "w2", "a blue square",
                       [["blue", "square"]]),
    ]
    generations = []
    for case_id, world_id, target, objects in (
            ("a", "w1", "a red circle", [["red", "circle"]]),
            ("b", "w2", "a blue square", [["blue", "square"]])):
        generations.append({
            "control": "correct", "case_id": case_id, "world_id": world_id,
            "generated_text": target, "target_text": target,
            "target_visible_objects": objects, "response_loss": 0.1,
        })
        generations.append({
            "control": "counterfactual", "case_id": case_id,
            "world_id": world_id,
            "generated_text": "a red circle",
            "target_text": "a green triangle",
            "target_visible_objects": [["green", "triangle"]],
            "response_loss": 0.4,
        })
        generations.append({
            "control": "shuffled", "case_id": case_id,
            "world_id": world_id, "generated_text": target,
            "target_text": target, "target_visible_objects": objects,
            "response_loss": 0.8,
        })
        generations.append({
            "control": "noimage", "case_id": case_id,
            "world_id": world_id, "generated_text": "unknown",
            "target_text": target, "target_visible_objects": objects,
            "response_loss": 1.2,
        })
    score = vba.score_caption_controls(
        labels, generations,
        controls=["correct", "counterfactual", "shuffled", "noimage"],
        seed=7, bootstrap={"cluster": "world", "confidence": 0.95,
                           "resamples": 100})
    assert score["controls"]["correct"]["token_f1"] == pytest.approx(1.0)
    assert score["controls"]["correct"]["semantic_f1"] == pytest.approx(1.0)
    assert score["dependence"]["counterfactual"]["token_f1_margin"] \
        > 0.0
    assert score["dependence"]["shuffled"]["loss_margin"] > 0.0
    assert score["dependence"]["noimage"]["loss_margin"] > 0.0
    broken = [dict(row) for row in generations]
    broken[0].pop("response_loss")
    with pytest.raises(ValueError):
        vba.score_caption_controls(
            labels, broken,
            controls=["correct", "counterfactual", "shuffled", "noimage"],
            seed=7, bootstrap={"cluster": "world", "confidence": 0.95,
                               "resamples": 100})


def _answer_case(case_id, world_id, label, task="color"):
    return {"case_id": case_id, "world_id": world_id, "task": task,
            "kind": None, "label": label}


def _answer_rows(case, generated):
    return [{"control": control, "case_id": case["case_id"],
             "generated_text": generated[control]} for control in generated]


def test_counterfactual_answer_scoring_answerable_scope():
    answerable = _answer_case("a", "w1", "red")
    abstain = _answer_case("b", "w2", "unknown", task="abstain")
    cases = [answerable, abstain]
    rows = []
    rows += _answer_rows(answerable,
                         {"correct": "red", "counterfactual": "blue",
                          "shuffled": "green", "noimage": "unknown"})
    rows += _answer_rows(abstain,
                         {"correct": "unknown", "counterfactual": "unknown",
                          "shuffled": "unknown", "noimage": "unknown"})
    score = vba.score_counterfactual_answers(
        cases, rows,
        controls=["correct", "counterfactual", "shuffled", "noimage"],
        seed=3, bootstrap={"cluster": "world", "confidence": 0.95,
                           "resamples": 100})
    assert score["counterfactual_answerable"]["cases"] == 1
    assert score["counterfactual_answerable"]["correct_margin"] == 1.0
    assert score["counterfactual_answerable"]["abstain_cases"] == 1
    assert score["counterfactual_answerable"][
        "abstain_counterfactual_unknown_rate"] == 1.0
    assert score["controls"]["correct"]["accuracy"] == 1.0


def test_protocol_gates_and_prepared_tamper_detection(tmp_path):
    built = _built()
    config = copy.deepcopy(_config())
    config["data_root"] = "prepared"
    parent = json.loads(
        (ROOT / config["parent_protocol"]["path"]).read_text())
    protocol = vba.build_protocol(config, built, parent, project_root=ROOT)
    assert protocol["claims"]["promotion"] is False
    assert protocol["claims"]["semantic_quality"] is False
    assert protocol["train"]["optimizer"] == "adamw_fused"
    assert protocol["train"]["fused"] is True
    assert protocol["schedule"]["total_updates"] == 10
    assert protocol["schedule"]["total_examples"] == 60
    assert protocol["trainable_scope"]["tensor_count"] == 6
    assert protocol["trainable_scope"]["backbone"] == "frozen"
    assert protocol["stopping"]["early_stopping"] is False
    assert protocol["limits"]["max_tool_calls"] == 0
    assert protocol["limits"]["max_rollout_steps"] == 1
    assert protocol["evaluation"]["heldout_answer_controls"] == [
        "correct", "wrong", "counterfactual", "shuffled", "noimage"]
    assert protocol["evaluation"]["gates"]["min_correct_accuracy"] \
        == config["evaluation"]["min_correct_accuracy"]
    assert protocol["resource_policy"]["downloads"] == 0
    assert protocol["checkpoint_policy"]["promotion"] is False
    vba.write_dataset(tmp_path / "prepared", built)
    protocol["data"]["root"] = str(tmp_path / "prepared")
    sha = vba.write_protocol(tmp_path / "prepared/protocol.json", protocol)
    assert len(sha) == 64
    loaded, manifest, implementation, data_root = vba.load_prepared(
        tmp_path / "prepared/protocol.json", ROOT)
    assert loaded["data"]["counts"] == built["manifest"]["counts"]
    assert data_root == tmp_path / "prepared"
    target = tmp_path / "prepared/evaluation/heldout_answer_cases.jsonl"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError):
        vba.load_prepared(tmp_path / "prepared/protocol.json", ROOT)


def test_gate_report_labels_diagnostic_only():
    evaluation = copy.deepcopy(_config()["evaluation"])
    answer = {
        "controls": {
            "correct": {"available": True, "accuracy": 0.9},
            "wrong": {"available": True}, "shuffled": {"available": True},
            "noimage": {"available": True},
        },
        "paired": {"wrong": {"correct_margin": 0.5},
                   "shuffled": {"correct_margin": 0.5},
                   "noimage": {"correct_margin": 0.5}},
        "counterfactual_answerable": {"correct_margin": 0.4},
        "unknown": {"abstention_accuracy": 0.9,
                    "false_answer_on_abstain": 0.1},
    }
    caption = {
        "controls": {"correct": {"token_f1": 0.8, "semantic_f1": 0.9}},
        "dependence": {
            "counterfactual": {"token_f1_margin": 0.5, "loss_margin": 0.5},
            "shuffled": {"token_f1_margin": 0.5, "loss_margin": 0.5},
            "noimage": {"token_f1_margin": 0.5, "loss_margin": 0.5},
        },
    }
    report = vba.diagnostic_gate_report(
        label="test", answer=answer, caption=caption,
        evaluation=evaluation)
    assert report["gate_failures"] == []
    assert report["diagnostic_only"] is True
    assert report["promotion"] is False
    answer["counterfactual_answerable"]["correct_margin"] = 0.0
    report = vba.diagnostic_gate_report(
        label="test", answer=answer, caption=caption,
        evaluation=evaluation)
    assert "min_counterfactual_margin_answerable" in report["gate_failures"]


def test_reload_artifact_comparison_and_serialization(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for directory in (left, right):
        (directory / "raw.jsonl").write_text('{"a":1}\n')
    assert all(vba.compare_reload_artifacts(
        left, right, ["raw.jsonl"]).values())
    (right / "raw.jsonl").write_text('{"a":2}\n')
    comparisons = vba.compare_reload_artifacts(left, right, ["raw.jsonl"])
    assert comparisons == {"raw.jsonl": False}
    protocol = {"format": vba.PROTOCOL_FORMAT, "value": 1}
    path = tmp_path / "protocol.json"
    vba.write_protocol(path, protocol)
    assert json.loads(path.read_text()) == protocol


def test_response_label_mask_and_sequence_budget():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "vision_balanced_runner",
        ROOT / "scripts/run_vision_balanced_alignment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from run_vision_curriculum import _example_tensors

    class DummyTokenizer:
        eos_token_id = 2

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [3 + (ord(char) % 7)
                                  for char in str(text)][:4]}

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False):
            return messages[0]["content"]

    class DummySession:
        device = "cpu"

    row = {"question": "What colour is the target?",
           "response": "red"}
    prompt, input_ids, labels = _example_tensors(
        DummySession(), DummyTokenizer(), row,
        maximum_prompt=64, maximum_sequence=64)
    prompt_length = prompt.shape[1]
    assert bool((labels[:, :prompt_length] == -100).all())
    assert bool((labels[:, prompt_length:] != -100).all())
    assert input_ids.shape[1] == labels.shape[1] == prompt_length + 4
    assert int(labels[0, -1]) == DummyTokenizer.eos_token_id
    with pytest.raises(Exception):
        _example_tensors(
            DummySession(), DummyTokenizer(), row,
            maximum_prompt=64, maximum_sequence=prompt_length + 1)


def test_cli_accepts_positional_and_mode_flag():
    spec = importlib.util.spec_from_file_location(
        "vision_balanced_runner_cli",
        ROOT / "scripts/run_vision_balanced_alignment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    common = ["--protocol", "p", "--device", "mps", "--output", "o",
              "--model-path", "m", "--model-lock", "l"]
    positional = module.build_parser().parse_args(["train", *common])
    flagged = module.build_parser().parse_args(["--mode", "reload", *common])
    assert positional.mode == "train" and positional.mode_flag is None
    assert flagged.mode is None and flagged.mode_flag == "reload"


def test_runner_imports_without_model():
    spec = importlib.util.spec_from_file_location(
        "vision_balanced_runner_import",
        ROOT / "scripts/run_vision_balanced_alignment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = module.run_updates(
        None, None, None, None,
        {"train": {"max_seconds": 1}, "limits": {}}, None, [],
        None, None, start_time=time.monotonic())
    assert records == []
