"""Model-free mutation tests for the independently scored native preflight."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from replica_lab.capacity_guard import CapacityGuard, CapacityPolicy
from replica_lab.episodes import input_view
from replica_lab.infrastructure_solver import solve_episode

REAL_ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, REAL_ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = script("score_infrastructure_generations")
GENERATOR = script("make_infrastructure_slice")
BACKEND_PINS = ("scripts/evaluate_text.py", "scripts/train_text.py", "src/replica_lab/model_adapter.py",
                "src/replica_lab/generation.py", "src/replica_lab/neural.py", "src/replica_lab/evaluation.py")
GUARD_PINS = ("scripts/run_generation_preflight.py", "scripts/evaluate_text.py", "src/replica_lab/capacity_guard.py")


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False) + "\n")


def dump_lines(path, values):
    Path(path).write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


class Tokenizer:
    eos_token_id = 1

    def get_chat_template(self):
        return "unit-test-template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<USER>" + messages[0]["content"] + "<ASSISTANT>"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        if len(text) > 64:
            return [1000 + int(hashlib.sha256(text.encode()).hexdigest()[:6], 16) % 100000, 250]
        return [ord(character) + 10 for character in text]

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False}
        return "".join("[EOS]" if value == 1 else chr(value - 10) for value in ids)


@pytest.fixture(scope="module")
def base_release(tmp_path_factory):
    path = tmp_path_factory.mktemp("score-source") / "release"
    GENERATOR.build_release(path)
    return path


@pytest.fixture
def bundle(tmp_path, monkeypatch, base_release):
    root = tmp_path / "project"
    root.mkdir()
    release = tmp_path / "release"
    shutil.copytree(base_release, release)
    monkeypatch.setattr(SCORER, "ROOT", root)
    pins = set(SCORER.INDEPENDENT_PINS) | set(BACKEND_PINS) | set(GUARD_PINS)
    for name in pins | {"scripts/validate_d2_experiment.py"}:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_ROOT / name, target)
    implementations = {name: SCORER.digest(root / name) for name in pins}
    model = root / "checkpoints/minicpm5_source"
    model.mkdir(parents=True)
    dump(model / "config.json", {"vocab_size": 130560})
    dump(model / "generation_config.json", {"eos_token_id": [1]})
    (model / "chat_template.jinja").write_text("unit-test-template")
    source_hashes = {name: SCORER.digest(model / name) for name in ("config.json", "generation_config.json", "chat_template.jinja")}
    lock = {"repo_id": SCORER.SOURCE_REPO, "revision": SCORER.SOURCE_REVISION,
            "local_bytes_sealed": True, "snapshot_files_sha256": source_hashes,
            "config_sha256": source_hashes["config.json"]}
    lock_path = root / "configs/minicpm5_sealed_lock.local.json"
    dump(lock_path, lock)
    canonical = SCORER.lines(release / "canonical/dev_episodes.jsonl")
    sources = {value["source_id"]: value for value in SCORER.read(release / "source_manifest.json")}
    cases, rows, examples = [], [], []
    tokenizer = Tokenizer()
    wanted = {"late_correction": "answer", "late_observation_ask": "ask",
              "late_observation_unknown": "answer", "tombstone": "unknown"}
    for family in SCORER.FAMILIES:
        for scenario, action in wanted.items():
            episode = next(value for value in canonical
                           if sources[value["sources"][0]["source_id"]]["family"] == family
                           and sources[value["sources"][0]["source_id"]]["scenario"] == scenario
                           and value["target"]["action"] == action)
            source = sources[episode["sources"][0]["source_id"]]
            target = solve_episode(episode, release, source)
            cases.append({"id": episode["id"], "world_id": source["world_id"], "family": family,
                          "scenario": scenario, "expected_action": target["action"], "expected_response": target["response"]})
            row = {"id": episode["id"], "group_id": episode["group_id"], "split": "dev",
                   **input_view(episode, purpose="local_evaluation"), "response": target["response"], "verified": True}
            rows.append(row)
            prompt = SCORER.prompt_ids(row, tokenizer, 512)
            generated = tokenizer.encode(row["response"], add_special_tokens=False) + [1]
            examples.append({"id": row["id"], "group_id": row["group_id"], "source_ids": row["source_ids"],
                "expected": row["response"], "prompt_tokens": len(prompt),
                "prompt_token_ids_sha256": hashlib.sha256(json.dumps([prompt]).encode()).hexdigest(),
                "generated_token_ids": generated, "generated_tokens": len(generated),
                "ended_by_eos": True, "hit_token_budget": False,
                "raw_completion": tokenizer.decode(generated, skip_special_tokens=False), "completion": row["response"],
                "elapsed_seconds": 0.5, "task_correct": False})  # deliberately wrong legacy score
    data = root / "selected_dev.jsonl"
    dump_lines(data, rows)
    specification = {"schema_version": "replica.infrastructure_preflight.v2", "release_path": str(release),
        "data_path": str(data), "data_sha256": SCORER.digest(data),
        "release_manifest_sha256": SCORER.digest(release / "manifest.json"),
        "source_manifest_sha256": SCORER.digest(release / "source_manifest.json"), "selected_cases": cases,
        "max_new_tokens": 32, "max_length": 512, "seed": 17, "decoding": "greedy",
        "model_path": str(model), "model_lock": str(lock_path), "implementation_sha256": implementations,
        "scope": "exposed fixtures"}
    spec = root / "spec.json"
    dump(spec, specification)
    directory = tmp_path / "native_run"
    directory.mkdir()
    report = {"status": "completed_native_generation_preflight", "device": "mps", "parameter_dtype": "float32",
        "compute_dtype": "float32", "parameter_dtypes": ["torch.float32"], "parameter_devices": ["mps"],
        "parameter_count": 2516756480, "max_examples": 16, "evaluated_examples": 16, "max_new_tokens": 32,
        "max_length": 512, "seed": 17, "evaluated_split": "dev", "data_path": str(data), "data_sha256": SCORER.digest(data),
        "model_path": str(model), "tokenizer_path": str(model), "model_lock_path": str(lock_path), "model_lock": lock,
        "effective_chat_template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
        "implementation_sha256": {name: implementations[name] for name in BACKEND_PINS},
        "native_use_cache": True, "decoding": {"do_sample": False, "num_beams": 1, "enable_thinking": False, "use_cache": True},
        "generation_config": {"do_sample": False, "num_beams": 1, "max_new_tokens": 32, "use_cache": True,
                              "stop_strings": None, "eos_token_id": [1]},
        "examples": examples, "fixture_task_accuracy": 0.0, "task_correct": 0, "evaluation_role": "legacy XOR description"}
    report_path, generations = directory / "generation_report.json", directory / "generations.jsonl"
    dump(report_path, report)
    dump_lines(generations, examples)
    return {"release": release, "spec": spec, "report_path": report_path, "generations": generations,
            "specification": specification, "report": report, "tokenizer": tokenizer, "root": root}


def score(bundle, **kwargs):
    return SCORER.score_generation(bundle["release"], bundle["spec"], bundle["report_path"],
                                   bundle["generations"], tokenizer=bundle["tokenizer"], **kwargs)


def rewrite_report(bundle, *, jsonl=True):
    dump(bundle["report_path"], bundle["report"])
    if jsonl:
        dump_lines(bundle["generations"], bundle["report"]["examples"])


def test_full_scorer_recomputes_truth_ignores_legacy_scores_and_reports_all_strata(bundle):
    result = score(bundle)
    assert result["overall"]["examples"] == 16
    assert result["overall"]["task_correct"] == {"count": 16, "rate": 1.0}
    assert result["overall"]["ended_by_eos"]["count"] == 16
    assert set(result["by_family"]) == set(SCORER.FAMILIES)
    assert {key: value["examples"] for key, value in result["by_action"].items()} == {"answer": 8, "ask": 4, "unknown": 4}
    assert result["guard_status"] == "not_supplied"
    assert not result["source_model"]["native_weights_rehashed"]
    assert result["examples"][0]["raw_completion"].endswith("[EOS]")


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "unknown"])
def test_exact_unique_complete_sixteen_ids_required_in_both_raw_artifacts(bundle, mutation):
    values = copy.deepcopy(bundle["report"]["examples"])
    if mutation == "duplicate":
        values[-1] = values[0]
    elif mutation == "missing":
        values.pop()
    else:
        values[-1]["id"] = "not-in-selected-dev"
    dump_lines(bundle["generations"], values)
    with pytest.raises(ValueError, match="Duplicate IDs|exactly the same"):
        score(bundle)


def test_jsonl_and_report_completions_must_match_even_if_individually_well_formed(bundle):
    bundle["report"]["examples"][0]["completion"] = "different"
    rewrite_report(bundle, jsonl=False)
    with pytest.raises(ValueError, match="JSONL differs"):
        score(bundle)


@pytest.mark.parametrize("mutation", ["wrong-text", "wrong-eos", "wrong-cap", "early-eos", "wrong-prompt"])
def test_token_text_endings_and_prompt_bindings_are_rechecked(bundle, mutation):
    example = bundle["report"]["examples"][0]
    if mutation == "wrong-text":
        example["completion"] = "forged correct text"
    elif mutation == "wrong-eos":
        example["ended_by_eos"] = False
    elif mutation == "wrong-cap":
        example["hit_token_budget"] = True
    elif mutation == "early-eos":
        example["generated_token_ids"].insert(0, 1)
        example["generated_tokens"] += 1
    else:
        example["prompt_token_ids_sha256"] = "0" * 64
    rewrite_report(bundle)
    with pytest.raises(ValueError, match="Completion text|EOS/cap|prompt hash"):
        score(bundle)


def test_capped_junk_is_preserved_as_format_failure_not_answer_extracted(bundle):
    example = bundle["report"]["examples"][0]
    text = "x" * 32
    example.update(generated_token_ids=bundle["tokenizer"].encode(text, add_special_tokens=False),
        generated_tokens=32, ended_by_eos=False, hit_token_budget=True, completion=text, raw_completion=text)
    rewrite_report(bundle)
    result = score(bundle)
    assert result["overall"]["task_correct"]["count"] == 15
    assert result["overall"]["format_valid"]["count"] == 15
    assert result["overall"]["hit_token_budget"]["count"] == 1
    assert result["examples"][0]["completion"] == text


@pytest.mark.parametrize("mutation", ["cpu", "half", "cap", "seed", "sampling", "export", "source-lock"])
def test_actual_mps_fp32_greedy_budget_and_original_source_are_required(bundle, mutation):
    report = bundle["report"]
    if mutation == "cpu":
        report["device"] = "cpu"
    elif mutation == "half":
        report["parameter_dtype"] = "float16"
    elif mutation == "cap":
        report["max_new_tokens"] = 256
    elif mutation == "seed":
        report["seed"] = 99
    elif mutation == "sampling":
        report["generation_config"]["do_sample"] = True
    elif mutation == "export":
        report["export_provenance"] = {"regime": "full"}
    else:
        report["model_lock"] = {**report["model_lock"], "revision": "wrong-revision"}
    rewrite_report(bundle)
    with pytest.raises(ValueError, match="MPS FP32|scope/budget/seed|configuration|frozen original|model lock"):
        score(bundle)


def test_spec_label_and_rehashed_input_target_do_not_override_independent_source_truth(bundle):
    bundle["specification"]["selected_cases"][0]["expected_response"] = "forged"
    dump(bundle["spec"], bundle["specification"])
    with pytest.raises(ValueError, match="independent raw-source truth"):
        score(bundle)


def test_partial_backend_provenance_is_rejected_even_when_remaining_hashes_match(bundle):
    bundle["report"]["implementation_sha256"].pop("scripts/train_text.py")
    rewrite_report(bundle)
    with pytest.raises(ValueError, match="backend implementation"):
        score(bundle)


@pytest.mark.parametrize("mutation", ["data", "release", "implementation", "tokenizer"])
def test_byte_pins_are_validated_instead_of_trusting_receipt_flags(bundle, mutation):
    if mutation == "data":
        Path(bundle["specification"]["data_path"]).write_text("corrupt\n")
    elif mutation == "release":
        source = next((bundle["release"] / "raw_sources").glob("*.jsonl"))
        source.write_bytes(source.read_bytes() + b"\n")
    elif mutation == "implementation":
        (bundle["root"] / "src/replica_lab/infrastructure_scoring.py").write_text("changed")
    else:
        (bundle["root"] / "checkpoints/minicpm5_source/chat_template.jinja").write_text("changed")
    with pytest.raises(ValueError, match="data bytes|Release file bytes|Implementation bytes|Sealed source"):
        score(bundle)


def test_optional_guard_is_exactly_replayed_and_offline_fallback_flags_are_checked(bundle):
    directory = bundle["report_path"].parent
    baseline = {"swap_used_bytes": 0, "memory_free_percent": 90.0}
    host = {"swap_used_bytes": 0, "memory_free_percent": 80.0}
    profile = {"type": "sample", "mps_driver_bytes": 1024}
    guard = CapacityGuard(CapacityPolicy(), baseline)
    guard.observe_host(host)
    guard.observe_profile(profile)
    host_path = directory.parent / "guard.host.jsonl"
    dump_lines(host_path, [host])
    dump_lines(directory / "profile_samples.jsonl", [profile])
    report = {"output": str(directory), "host_samples": str(host_path), "guard": guard.report(),
              "host_poll_seconds": 2, "profile_poll_seconds": 0.25,
              "command": [sys.executable, str(bundle["root"] / "scripts/evaluate_text.py"), "--output", str(directory)],
              "implementation_sha256": {name: bundle["specification"]["implementation_sha256"][name] for name in GUARD_PINS},
              "status": "completed_within_declared_sampled_limits", "child_exit_code": 0,
              "offline_environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                  "HF_HUB_DISABLE_TELEMETRY": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}}
    guard_path = directory.parent / "guard.json"
    dump(guard_path, report)
    assert score(bundle, guard_path=guard_path)["guard_replay"]["exact_replay"]
    report["offline_environment"]["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    dump(guard_path, report)
    with pytest.raises(ValueError, match="offline MPS without fallback"):
        score(bundle, guard_path=guard_path)
