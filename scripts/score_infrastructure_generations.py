#!/usr/bin/env python3
"""Independently score the fixed v2 native generation preflight, offline on CPU.

The legacy generator's scores and XOR descriptions are deliberately ignored.
No model is instantiated. Only the sealed local tokenizer is loaded to verify
that archived token IDs, whole completions, prompt hashes and endings agree.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.episodes import input_view
from replica_lab.infrastructure_scoring import SCORING_VERSION, score_completion
from replica_lab.infrastructure_solver import FAMILIES, SOLVER_VERSION, solve_episode

VERSION = "replica.infrastructure_generation_score.v2"
SOURCE_REPO = "openbmb/MiniCPM5-2B"
SOURCE_REVISION = "f97400052a43d642bbc6e9975e2397e3ae6a6b52"
INDEPENDENT_PINS = {
    "scripts/score_infrastructure_generations.py", "src/replica_lab/infrastructure_solver.py",
    "src/replica_lab/infrastructure_scoring.py", "src/replica_lab/episodes.py",
    "src/replica_lab/contracts.py", "configs/infrastructure_slice_v2_design.json"}
BACKEND_PINS = {"scripts/evaluate_text.py", "scripts/train_text.py", "src/replica_lab/model_adapter.py",
                "src/replica_lab/generation.py", "src/replica_lab/neural.py", "src/replica_lab/evaluation.py"}
DIMENSIONS = ("format_valid", "action_correct", "decision_correct", "value_correct",
              "task_correct", "exact_text_match", "ended_by_eos", "hit_token_budget")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def safe_file(root, name):
    root, relative = Path(root).resolve(), Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Pinned path escapes its root")
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root), "Pinned path escapes its root")
    return path


def check_pins(pins, *, required=()):
    require(isinstance(pins, dict) and bool(pins) and set(required).issubset(pins),
            "Missing required implementation pins")
    for name, expected in pins.items():
        require(digest(safe_file(ROOT, name)) == expected, f"Implementation bytes changed: {name}")


def unique(values, name):
    require(isinstance(values, list) and all(isinstance(value, dict) and isinstance(value.get("id"), str)
                                           for value in values), f"Malformed {name} inventory")
    result = {value["id"]: value for value in values}
    require(len(result) == len(values), f"Duplicate IDs in {name}")
    return result


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)


def prompt_ids(row, tokenizer, max_length):
    """Rebuild the documented native prompt without importing the trainer."""
    memory = "\n".join(row["memory_texts"])
    content = "Observed source records:\n" + (memory or "[none]") + "\n\nTask:\n" + row["prompt"]
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer.encode(rendered, add_special_tokens=False)
    response = tokenizer.encode(row["response"], add_special_tokens=False)
    require(bool(prompt) and tokenizer.eos_token_id is not None
            and len(prompt) + len(response) + 1 <= max_length, "Reference row exceeds fixed sequence budget")
    return prompt


def summary(examples):
    count = len(examples)
    require(count > 0, "Cannot summarize empty stratum")
    return {"examples": count,
            **{key: {"count": sum(example[key] for example in examples),
                     "rate": sum(example[key] for example in examples) / count} for key in DIMENSIONS},
            "generated_tokens": sum(example["generated_tokens"] for example in examples),
            "generation_seconds": sum(example["elapsed_seconds"] for example in examples)}


def grouped(examples, key):
    return {value: summary([example for example in examples if example[key] == value])
            for value in sorted({example[key] for example in examples})}


def _source_lock(spec, report):
    model_path, lock_path = resolved(spec["model_path"]), resolved(spec["model_lock"])
    require(model_path == (ROOT / "checkpoints/minicpm5_source").resolve()
            and lock_path == (ROOT / "configs/minicpm5_sealed_lock.local.json").resolve(),
            "Preflight must reuse the original local source and sealed lock")
    require(resolved(report["model_path"]) == model_path and resolved(report["tokenizer_path"]) == model_path
            and resolved(report["model_lock_path"]) == lock_path and "export_provenance" not in report,
            "Generation report is not the declared frozen original source")
    lock = read(lock_path)
    require(report["model_lock"] == lock and lock.get("local_bytes_sealed") is True
            and lock.get("repo_id") == SOURCE_REPO and lock.get("revision") == SOURCE_REVISION,
            "Source model lock differs")
    small_hashes = {}
    for name, expected in lock["snapshot_files_sha256"].items():
        if name.endswith(".safetensors"):
            continue  # The launch seal and retained source receipt cover native weights.
        path = safe_file(model_path, name)
        actual = digest(path)
        require(actual == expected, f"Sealed source tokenizer/config bytes changed: {name}")
        small_hashes[name] = actual
    config = read(model_path / "config.json")
    generation = read(model_path / "generation_config.json")
    require(digest(model_path / "config.json") == lock["config_sha256"], "Source config differs from lock")
    require(generation["eos_token_id"] == report["generation_config"]["eos_token_id"],
            "Generation EOS configuration differs from sealed source")
    return model_path, lock_path, lock, config, small_hashes


def _metadata(spec, report, data_path):
    require(spec["schema_version"] == "replica.infrastructure_preflight.v2", "Unsupported preflight specification")
    require(spec["max_new_tokens"] == 32 and spec["max_length"] == 512 and spec["seed"] == 17,
            "Preflight specification changes fixed budgets or seed")
    decoding = spec["decoding"]
    require(decoding == "greedy" or (isinstance(decoding, dict) and decoding.get("do_sample") is False
            and decoding.get("num_beams") == 1 and decoding.get("enable_thinking") is False),
            "Specification must declare fixed greedy decoding with thinking disabled")
    require(report["status"] == "completed_native_generation_preflight" and report["device"] == "mps"
            and report["parameter_dtype"] == report["compute_dtype"] == "float32"
            and report["parameter_dtypes"] == ["torch.float32"] and report["parameter_devices"] == ["mps"]
            and report["parameter_count"] == 2516756480, "Generation did not use full native MPS FP32")
    require(report["max_examples"] == report["evaluated_examples"] == 16
            and report["max_new_tokens"] == 32 and report["max_length"] == 512 and report["seed"] == 17
            and report["evaluated_split"] == "dev", "Generation scope/budget/seed differs")
    require(resolved(report["data_path"]) == data_path and report["data_sha256"] == spec["data_sha256"],
            "Generation used a different data file")
    require(report["decoding"].get("do_sample") is False and report["decoding"].get("num_beams") == 1
            and report["decoding"].get("enable_thinking") is False and report["decoding"].get("use_cache") is True
            and report["native_use_cache"] is True, "Native decoding differs from frozen greedy contract")
    require(report["generation_config"].get("do_sample") is False
            and report["generation_config"].get("num_beams") == 1
            and report["generation_config"].get("max_new_tokens") == 32
            and report["generation_config"].get("use_cache") is True
            and report["generation_config"].get("stop_strings") is None,
            "Effective generation configuration differs")


def _guard(directory, guard_path, spec):
    guard_report = read(guard_path)
    offline = guard_report["offline_environment"]
    require(all(offline.get(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"))
            and offline.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0", "Guard did not enforce offline MPS without fallback")
    require(all(spec["implementation_sha256"].get(name) == value
                for name, value in guard_report["implementation_sha256"].items()),
            "Guard implementation differs from preflight pins")
    module_spec = importlib.util.spec_from_file_location("infrastructure_guard_replay", ROOT / "scripts/validate_d2_experiment.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.replay_guard(directory, guard_path=guard_path, require_native=True)


def score_generation(release, spec_path, report_path, generations_path, *, guard_path=None, tokenizer=None):
    """Validate and score a complete fixed preflight; model-free and read-only."""
    release, spec_path = resolved(release), resolved(spec_path)
    report_path, generations_path = resolved(report_path), resolved(generations_path)
    spec, report = read(spec_path), read(report_path)
    require(resolved(spec["release_path"]) == release, "Specification names a different release")
    require(digest(release / "manifest.json") == spec["release_manifest_sha256"]
            and digest(release / "source_manifest.json") == spec["source_manifest_sha256"], "Release/source manifest pin differs")
    manifest = read(release / "manifest.json")
    require(manifest["schema_version"] == "replica.infrastructure_slice.v2" and manifest["seed"] == 41
            and manifest["canonical_target_count"] == 2000 and manifest["world_count"] == 1000,
            "Release is not the preregistered 2,000-target infrastructure slice")
    for name, expected in manifest["files_sha256"].items():
        require(digest(safe_file(release, name)) == expected, f"Release file bytes changed: {name}")
    check_pins(spec["implementation_sha256"], required=INDEPENDENT_PINS)
    backend_pins = report["implementation_sha256"]
    require(set(backend_pins) == BACKEND_PINS and all(spec["implementation_sha256"].get(name) == value
            for name, value in backend_pins.items()), "Native backend implementation differs from specification")
    data_path = resolved(spec["data_path"])
    require(digest(data_path) == spec["data_sha256"], "Selected data bytes changed")
    _metadata(spec, report, data_path)
    model_path, lock_path, lock, config, small_hashes = _source_lock(spec, report)
    tokenizer = load_tokenizer(model_path) if tokenizer is None else tokenizer
    template_hash = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    require(report["effective_chat_template_sha256"] == template_hash, "Effective chat template differs")
    eos = report["generation_config"]["eos_token_id"]
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    selected = unique(spec["selected_cases"], "specification")
    rows = unique(lines(data_path), "selected data")
    canonical = unique(lines(release / "canonical/dev_episodes.jsonl"), "canonical dev")
    report_examples = unique(report["examples"], "generation report")
    raw_examples = unique(lines(generations_path), "generation JSONL")
    ids = set(selected)
    require(len(ids) == 16 and ids == set(rows) == set(report_examples) == set(raw_examples),
            "Selected/report/JSONL IDs must cover exactly the same sixteen cases")
    require(set(selected).issubset(canonical), "Selected case is not canonical development data")
    require(len({value["world_id"] for value in selected.values()}) == 16,
            "Preflight requires sixteen independent source worlds")
    sources = read(release / "source_manifest.json")
    source_map = {value["source_id"]: value for value in sources}
    require(len(source_map) == len(sources), "Duplicate source manifest identities")
    examples = []
    for case in spec["selected_cases"]:
        identity = case["id"]
        episode, row = canonical[identity], rows[identity]
        require(episode["split"] == "dev" and len(episode["sources"]) == 1, "Selected canonical role/source differs")
        source = source_map[episode["sources"][0]["source_id"]]
        expected = solve_episode(episode, release, source)
        require(expected == episode["target"] and all(case[key] == source[key] for key in ("world_id", "family", "scenario"))
                and case["expected_action"] == expected["action"] and case["expected_response"] == expected["response"],
                "Selected expectation differs from independent raw-source truth")
        view = input_view(episode, purpose="local_evaluation")
        expected_row = {"id": identity, "group_id": episode["group_id"], "split": "dev", **view,
                        "response": expected["response"], "verified": True}
        require(row == expected_row, "Selected data differs from independently verified canonical export")
        example = report_examples[identity]
        require(example == raw_examples[identity], "Generation JSONL differs from report example")
        require(example["group_id"] == row["group_id"] and example["source_ids"] == row["source_ids"]
                and example["expected"] == expected["response"], "Generated case provenance/expected text differs")
        token_ids = example["generated_token_ids"]
        require(isinstance(token_ids, list) and 0 < len(token_ids) <= 32
                and all(type(token) is int and 0 <= token < config["vocab_size"] for token in token_ids)
                and type(example["generated_tokens"]) is int and example["generated_tokens"] == len(token_ids),
                "Generated token IDs/count exceed fixed budget or vocabulary")
        ended, capped = example["ended_by_eos"], example["hit_token_budget"]
        require(type(ended) is bool and type(capped) is bool and ended == (token_ids[-1] in eos_ids)
                and not any(token in eos_ids for token in token_ids[:-1])
                and capped == (not ended and len(token_ids) == 32) and (ended or capped),
                "Generated EOS/cap metadata is incoherent")
        answer_ids = token_ids[:-1] if ended else token_ids
        require(example["raw_completion"] == tokenizer.decode(token_ids, skip_special_tokens=False)
                and example["completion"] == tokenizer.decode(answer_ids, skip_special_tokens=False),
                "Completion text differs from archived token IDs")
        prompt = prompt_ids(row, tokenizer, 512)
        require(example["prompt_tokens"] == len(prompt) and example["prompt_token_ids_sha256"]
                == hashlib.sha256(json.dumps([prompt]).encode()).hexdigest(), "Generated prompt hash/count differs")
        require(type(example["elapsed_seconds"]) in (int, float) and math.isfinite(example["elapsed_seconds"])
                and example["elapsed_seconds"] >= 0, "Invalid generation timing")
        scored = score_completion(example["completion"], expected, family=source["family"],
                                  ended_by_eos=ended, hit_token_budget=capped)
        examples.append({"id": identity, "world_id": source["world_id"], "family": source["family"],
                         "scenario": source["scenario"], "expected_action": expected["action"],
                         "expected_response": expected["response"], "support_event_ids": expected["support_event_ids"],
                         **{key: example[key] for key in ("group_id", "source_ids", "prompt_tokens",
                             "prompt_token_ids_sha256", "generated_token_ids", "generated_tokens", "raw_completion",
                             "completion", "elapsed_seconds")}, **scored})
    guard = _guard(report_path.parent, resolved(guard_path), spec) if guard_path is not None else None
    return {"schema_version": VERSION, "status": "completed_independent_infrastructure_scoring",
            "scope": "Sixteen exposed synthetic development cases; descriptive infrastructure preflight only.",
            "solver_version": SOLVER_VERSION, "scoring_version": SCORING_VERSION,
            "inputs_sha256": {"specification": digest(spec_path), "release_manifest": digest(release / "manifest.json"),
                "source_manifest": digest(release / "source_manifest.json"), "data": digest(data_path),
                "generation_report": digest(report_path), "generations": digest(generations_path), "model_lock": digest(lock_path)},
            "implementation_sha256": spec["implementation_sha256"], "source_model": {"repo_id": lock["repo_id"],
                "revision": lock["revision"], "path": str(model_path), "small_files_sha256": small_hashes,
                "native_weights_rehashed": False}, "template_sha256": template_hash,
            "guard_replay": guard, "guard_status": "exact_replay_passed" if guard else "not_supplied",
            "overall": summary(examples), "by_family": grouped(examples, "family"),
            "by_action": grouped(examples, "expected_action"), "by_scenario": grouped(examples, "scenario"),
            "response_histogram": dict(Counter(example["completion"] for example in examples)),
            "examples": examples,
            "legacy_fields_ignored": ["XOR-only descriptions", "fixture_task_accuracy", "task_correct aggregates",
                                      "legacy per-example correctness scores"],
            "limitations": ["No architecture comparison, calibration threshold, retention or sealed benchmark claim",
                            "All four mechanisms and question grammars are shared across release roles",
                            "Related queries are clustered by source world; selected cases use distinct worlds",
                            "Tool-recovery outputs are static recommendations; no tool operation was executed",
                            "Native weight source integrity is covered by launch seal and retained source evidence"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("release", "spec", "generation-report", "generations", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--guard", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Use a fresh output file to preserve earlier scoring evidence")
    result = score_generation(args.release, args.spec, args.generation_report, args.generations,
                              guard_path=args.guard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "examples": result["overall"]["examples"],
                      "task_correct": result["overall"]["task_correct"]["count"]}))


if __name__ == "__main__":
    main()
