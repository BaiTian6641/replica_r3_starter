"""Run real native tiny-Llama training through the resumable CLI engine on CPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from replica_lab import model_adapter
from replica_lab.checkpointing import validate_training_checkpoint


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("integration_train_text", ROOT / "scripts/train_text.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class SmallTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return "[CHAT]" + messages[0]["content"]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text in {"0", "1"}:
            return [3 + int(text)]
        if text.startswith("[CHAT]"):
            # Two distinct input prompts; gold answer strings never enter this
            # branch. A third input is reserved for the separate dev family.
            value = next(value for value in range(3) if f"Input {value}." in text)
            return [0, 7, 9 + value, 8]
        return [5, 6]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join({1: "</s>", 3: "0", 4: "1"}.get(value, f"<{value}>") for value in ids)

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text('{"version":"1.0"}\n')
        (directory / "tokenizer_config.json").write_text('{"eos_token":"</s>"}\n')


class PhaseRecorder:
    def __init__(self):
        self.phases = []

    def mark(self, phase, **fields):
        self.phases.append((phase, fields))


@pytest.fixture
def tiny_source(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SmallTokenizer())
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    source = tmp_path / "sealed-source"
    torch.manual_seed(29)
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=32,
                         attention_dropout=0.0, tie_word_embeddings=False,
                         bos_token_id=0, eos_token_id=1, pad_token_id=2)
    config._attn_implementation = "eager"
    native = LlamaForCausalLM(config)
    count = sum(parameter.numel() for parameter in native.parameters())
    native.save_pretrained(source, safe_serialization=True)
    SmallTokenizer().save_pretrained(source)
    digest = hashlib.sha256((source / "config.json").read_bytes()).hexdigest()
    lock = {"repo_id": "test/tiny-native", "revision": "sealed-tiny-revision",
            "config_sha256": digest}
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))

    def strict_tiny_lock(model_path, supplied_lock):
        assert Path(model_path) == source
        assert supplied_lock == lock
        actual = hashlib.sha256((Path(model_path) / "config.json").read_bytes()).hexdigest()
        assert actual == supplied_lock["config_sha256"]

    monkeypatch.setattr(model_adapter, "validate_model_lock", strict_tiny_lock)
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"

    def row(split, value, answer):
        return {"id": f"{split}-{value}", "group_id": f"{split}-source-family",
                "split": split, "prompt": f"Input {value}.", "response": str(answer),
                "memory_texts": [f"Observed {split} rule."],
                "source_ids": [f"fixture://{split}-source"], "verified": True}

    train.write_text("".join(json.dumps(row("train", value, value)) + "\n" for value in range(2)))
    dev.write_text(json.dumps(row("dev", 2, 0)) + "\n")
    yield SimpleNamespace(source=source, lock=lock_path, train=train, dev=dev,
                          count=count, root=tmp_path)
    torch.set_num_threads(previous_threads)


def arguments(source, output_name, *, stop_after=None, resume_from=None):
    output = source.root / output_name
    output.mkdir()
    return SimpleNamespace(
        model_path=source.source, model_lock=source.lock, data=source.train,
        dev=source.dev, output=output, regime="full", device="cpu", dtype="float32",
        optimizer="muon8bit", steps=2, grad_accum=2, max_length=32,
        lr_base=1e-4, lr_new=1e-4, lr_muon_base=1e-4, lr_muon_new=1e-4,
        momentum=0.95, ns_steps=5, block_size=64, weight_decay=0.01, seed=17,
        memory_size=8, workspace_size=8, insertion_layer=-1,
        checkpointing=True, verify_reload=True, sampling="answer_balanced",
        lr_schedule="constant", warmup_steps=2, min_lr_ratio=0.2,
        generation_every=1, generation_max_new_tokens=2,
        stop_on_train_perfect=False, checkpoint_every=1,
        resume_from=resume_from, stop_after=stop_after)


def execute(source, args):
    recorder = PhaseRecorder()
    report = TRAIN.execute(args, recorder, expected_original_parameters=source.count)
    return report, recorder


def test_tiny_native_runner_resume_matches_uninterrupted_weights_and_history(tiny_source):
    full_args = arguments(tiny_source, "uninterrupted")
    full, _ = execute(tiny_source, full_args)
    first_args = arguments(tiny_source, "first-chunk", stop_after=1)
    first, _ = execute(tiny_source, first_args)
    resume = first_args.output / "training_checkpoint_0001"
    second_args = arguments(tiny_source, "second-chunk", resume_from=resume)
    resumed, phases = execute(tiny_source, second_args)

    assert full["steps"] == resumed["steps"] == 2
    assert first["steps"] == 1 and first["stop_reason"] == "process_chunk_completed"
    assert resumed["start_step"] == 1 and resumed["stop_reason"] == "step_budget_completed"
    assert full["run_contract"] == first["run_contract"] == resumed["run_contract"]
    assert any(phase == "runtime_restored" and fields["completed_steps"] == 1
               for phase, fields in phases.phases)
    assert full["processed_train_tokens"] == resumed["processed_train_tokens"] == 24
    assert full["supervised_train_tokens"] == resumed["supervised_train_tokens"] == 8
    assert full["dev_supervised_tokens"] == resumed["dev_supervised_tokens"] == 2
    assert full["dev_response_nll"] == resumed["dev_response_nll"]
    assert full["dev_answer_nll"] == resumed["dev_answer_nll"]
    assert full["dev_eos_nll"] == resumed["dev_eos_nll"]

    # Every optimizer update must include both classes. Warmup is based on the
    # global step and original base rates, so resuming must not compound 0.5.
    for report in (full, resumed):
        assert [item["lr_factor"] for item in report["train_updates"]] == [0.5, 1.0]
        assert [item["step"] for item in report["generation_history"]] == [1, 2]
        assert report["reload_verification"]["status"] == "passed"
        assert report["reload_verification"]["maximum_absolute_logit_difference"] == 0.0
        assert report["optimizer"]["state_bytes_by_dtype"]["torch.int8"] > 0
        assert report["first_update_gradient_audit"]["original"]["missing_gradient_names"] == []
        assert report["parameters"]["original"]["trainable"] == tiny_source.count
        for index, item in enumerate(report["train_updates"]):
            assert item["selection"]["answer_histogram"] == {"0": 1, "1": 1}
            assert item["selection"]["all_answer_classes_present"]
            assert set(item["example_ids"]) == {"train-0", "train-1"}
            assert set(item["learning_rates"]) == {1e-4 * (0.5 if index == 0 else 1.0)}

    # The first chunk's histories survive the runtime checkpoint verbatim; only
    # wall time from the independently executed second update may differ.
    assert resumed["train_updates"][0] == first["train_updates"][0]
    for expected, actual in zip(full["train_updates"], resumed["train_updates"]):
        for key in ("step", "example_ids", "selection", "lr_factor", "learning_rates",
                    "answer_nll", "eos_nll", "train_response_nll_before_update"):
            assert actual[key] == expected[key]
    assert full["generation_history"] == resumed["generation_history"]

    expected_weights = load_file(full_args.output / "checkpoint/base/model.safetensors")
    actual_weights = load_file(second_args.output / "checkpoint/base/model.safetensors")
    assert expected_weights.keys() == actual_weights.keys()
    for name in expected_weights:
        torch.testing.assert_close(actual_weights[name], expected_weights[name], atol=0, rtol=0)

    final_checkpoint = second_args.output / "training_checkpoint_0002"
    receipt = validate_training_checkpoint(final_checkpoint, resumed["run_contract"])
    assert receipt["completed_steps"] == 2
    runtime = torch.load(final_checkpoint / "runtime.pt", map_location="cpu", weights_only=True)
    assert runtime["metadata"]["data_state"]["next_step"] == 2
    assert runtime["metadata"]["data_state"]["update_reports"] == resumed["train_updates"]
    assert runtime["metadata"]["schedule_state"]["base_rates"] == [1e-4, 1e-4]


def test_native_runner_keeps_production_parameter_guard_by_default(tiny_source):
    args = arguments(tiny_source, "wrong-model")
    with pytest.raises(RuntimeError, match="Unexpected unique parameter count"):
        TRAIN.execute(args, PhaseRecorder())


def test_modules_only_runner_opens_upstream_gradients_and_preserves_base(tiny_source):
    args = arguments(tiny_source, "d1-first", stop_after=1)
    args.regime = "modules_only"
    first, _ = execute(tiny_source, args)
    resumed_args = arguments(tiny_source, "d1-resumed",
                             resume_from=args.output / "training_checkpoint_0001")
    resumed_args.regime = "modules_only"
    report, _ = execute(tiny_source, resumed_args)
    assert report["parameters"]["original"]["trainable"] == 0
    assert first["first_update_gradient_audit"]["new"]["nonzero_gradient_tensors"] == 2
    assert report["first_update_gradient_audit"]["new"]["nonzero_gradient_tensors"] > 2
    assert report["first_update_gradient_audit"]["original"]["gradient_tensors"] == 0
    assert report["reload_verification"]["maximum_absolute_logit_difference"] == 0
    assert report["generation_history"][-1]["decoding"]["use_cache"] is False
    original = load_file(tiny_source.source / "model.safetensors")
    exported = load_file(resumed_args.output / "checkpoint/base/model.safetensors")
    assert original.keys() == exported.keys()
    for name in original:
        assert torch.equal(original[name], exported[name]), name
    initial = load_file(args.output / "checkpoint/sidecars.safetensors")
    final = load_file(resumed_args.output / "checkpoint/sidecars.safetensors")
    assert sum(not torch.equal(initial[name], final[name]) for name in initial) > 2


def test_full_augmented_runner_resumes_all_scopes_sidecars_and_generation(tiny_source):
    def d2_arguments(name, **kwargs):
        args = arguments(tiny_source, name, **kwargs)
        args.regime = "full_augmented"
        # Distinct original/new rates expose lost ownership or schedule state.
        args.lr_new = args.lr_muon_new = 3e-4
        return args

    continuous_args = d2_arguments("d2-uninterrupted")
    continuous, _ = execute(tiny_source, continuous_args)
    first_args = d2_arguments("d2-first", stop_after=1)
    first, _ = execute(tiny_source, first_args)
    resumed_args = d2_arguments("d2-resumed",
        resume_from=first_args.output / "training_checkpoint_0001")
    resumed, phases = execute(tiny_source, resumed_args)
    assert resumed["run_contract"] == first["run_contract"] == continuous["run_contract"]
    assert resumed["start_step"] == 1 and resumed["steps"] == continuous["steps"] == 2
    assert any(name == "runtime_restored" and fields["completed_steps"] == 1
               for name, fields in phases.phases)

    original = load_file(tiny_source.source / "model.safetensors")
    native = load_file(resumed_args.output / "checkpoint/base/model.safetensors")
    sidecars = load_file(resumed_args.output / "checkpoint/sidecars.safetensors")
    scope = resumed["parameters"]
    assert scope["regime"] == "full_augmented"
    assert scope["original"]["total"] == scope["original"]["trainable"] == tiny_source.count
    assert scope["new"]["trainable"] == scope["new"]["total"] > 0
    assert scope["new"]["tensor_count"] == len(sidecars) == 48

    groups = resumed["optimizer"]["groups"]
    assert [(group["scope"], group["algorithm"]) for group in groups] == [
        ("original", "muon8bit"), ("original", "adamw"),
        ("new", "muon8bit"), ("new", "adamw")]
    owned_names = [name for group in groups for name in group["param_names"]]
    # This fixture inserts at its only native decoder block. Native safetensor
    # names stay unwrapped, while live optimizer ownership includes .original.
    expected_original = {"base." + name.replace("model.layers.0.",
        "model.layers.0.original.", 1) for name in original}
    expected_new = set(sidecars)
    assert set(owned_names) == expected_original | expected_new
    assert len(owned_names) == len(set(owned_names)) == len(original) + 48
    for group in groups:
        expected_names = expected_original if group["scope"] == "original" else expected_new
        assert set(group["param_names"]).issubset(expected_names)
    assert sum(group["parameters"] for group in groups) == tiny_source.count + scope["new"]["total"]

    first_grad = first["first_update_gradient_audit"]
    later_grad = resumed["first_update_gradient_audit"]
    for audit in (first_grad, later_grad):
        assert audit["original"]["eligible_tensors"] == len(original)
        assert audit["original"]["gradient_tensors"] == len(original)
        assert audit["original"]["nonzero_gradient_tensors"] == len(original)
        assert not audit["original"]["missing_gradient_names"]
        assert audit["new"]["eligible_tensors"] == audit["new"]["gradient_tensors"] == 48
        assert not audit["new"]["missing_gradient_names"]
    assert first_grad["new"]["nonzero_gradient_tensors"] == 2
    assert later_grad["new"]["nonzero_gradient_tensors"] == 48
    assert original.keys() == native.keys()
    for name in original:
        assert not torch.equal(original[name], native[name]), name
    assert not torch.equal(original["model.embed_tokens.weight"], native["model.embed_tokens.weight"])
    assert not torch.equal(original["lm_head.weight"], native["lm_head.weight"])
    first_sidecars = load_file(first_args.output / "checkpoint/sidecars.safetensors")
    assert sum(not torch.equal(first_sidecars[name], sidecars[name]) for name in sidecars) > 2

    # Resume must reproduce the complete adapted model, including nonzero
    # branches, rather than only reloading matching original base weights.
    for relative in ("base/model.safetensors", "sidecars.safetensors"):
        expected = load_file(continuous_args.output / "checkpoint" / relative)
        actual = load_file(resumed_args.output / "checkpoint" / relative)
        assert expected.keys() == actual.keys()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert continuous["dev_response_nll"] == resumed["dev_response_nll"]
    assert continuous["generation_history"] == resumed["generation_history"]
    for report in (first, continuous, resumed):
        assert report["reload_verification"]["status"] == "passed"
        assert report["reload_verification"]["maximum_absolute_logit_difference"] == 0.0
        for check in report["generation_history"]:
            assert check["decoding"]["use_cache"] is False
            assert check["decoding"]["context"] == "fixed_original_prompt_and_support"
            assert check["evaluated_examples"] == 2
            assert all(example["prompt_tokens"] == 4 for example in check["examples"])
    assert [check["step"] for check in resumed["generation_history"]] == [1, 2]
    assert resumed["train_updates"][0] == first["train_updates"][0]
    assert [step["learning_rates"] for step in resumed["train_updates"]] == [
        [5e-5, 5e-5, 1.5e-4, 1.5e-4], [1e-4, 1e-4, 3e-4, 3e-4]]

    checkpoint = resumed_args.output / "training_checkpoint_0002"
    receipt = validate_training_checkpoint(checkpoint, resumed["run_contract"])
    assert receipt["completed_steps"] == 2
    assert "sidecars.safetensors" in receipt["files_sha256"]
    runtime = torch.load(checkpoint / "runtime.pt", map_location="cpu", weights_only=True)
    expected_runtime = torch.load(continuous_args.output / "training_checkpoint_0002/runtime.pt",
                                 map_location="cpu", weights_only=True)
    assert runtime["metadata"]["data_state"]["next_step"] == 2
    assert runtime["metadata"]["data_state"]["update_reports"] == resumed["train_updates"]
    assert runtime["metadata"]["schedule_state"] == {
        "completed_steps": 2, "base_rates": [1e-4, 1e-4, 3e-4, 3e-4]}
    assert runtime["optimizer"]["param_groups"] == expected_runtime["optimizer"]["param_groups"]
    assert runtime["optimizer"]["state"].keys() == expected_runtime["optimizer"]["state"].keys()
    for identity, state in runtime["optimizer"]["state"].items():
        for key, value in state.items():
            expected = expected_runtime["optimizer"]["state"][identity][key]
            if isinstance(value, torch.Tensor):
                assert value.dtype == expected.dtype
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            else:
                assert value == expected
    for name, value in load_file(checkpoint / "sidecars.safetensors").items():
        torch.testing.assert_close(value, sidecars[name], rtol=0, atol=0)
