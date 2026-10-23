#!/usr/bin/env python3
"""Recheck a declared response-only projection and pair it with the failed B1 run.

The model is never loaded. The sealed local tokenizer is used only to bind the
native prompt and output token IDs. Raw source replay determines every target.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("response_contract_parent_scorer", ROOT / "scripts/score_infrastructure_generations.py")
BASE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BASE)

from replica_lab.episodes import input_view
from replica_lab.contracts import canonical_json
from replica_lab.infrastructure_scoring import SCORING_VERSION, score_completion
from replica_lab.infrastructure_solver import SOLVER_VERSION, solve_episode

VERSION = "replica.response_contract_generation_score.v1"
INDEPENDENT_PINS = BASE.INDEPENDENT_PINS | {
    "scripts/score_response_contract_generations.py", "src/replica_lab/response_contract.py",
    "scripts/prepare_response_contract.py", "configs/response_contract_v1.json"}
require, digest, read, lines, resolved, safe_file = (
    BASE.require, BASE.digest, BASE.read, BASE.lines, BASE.resolved, BASE.safe_file)


def independent_instruction(family, contract):
    """Reconstruct static wording from declared data, independently of projection code."""
    require(contract["schema_version"] == "replica.response_contract.v1"
            and contract["status"] == "predeclared_before_native_generation"
            and contract["append_separator"] == "\n\n" and contract["joiner"] == "\n"
            and contract["projection"] == "canonical_prompt_plus_separator_plus_family_instruction_v1"
            and contract["projection_inputs"] == ["family", "static_contract_only"],
            "Unsupported static response contract")
    orders = {"register_xor8": ["RULE", "INPUT", "MASK"], "affine_mod97": ["RULE", "A", "X", "B"],
              "finite_state": ["RULE", "START", "SYMBOLS"], "tool_recovery": ["RULE", "STATUS", "ATTEMPT", "LIMIT"]}
    require(set(contract["family_formats"]) == set(orders) and family in orders,
            "Contract family inventory differs")
    formats = contract["family_formats"]
    require(all(value["field_order"] == orders[name] for name, value in formats.items()),
            "Contract field order differs from canonical grammar")
    return (contract["common_prefix"] + contract["joiner"]
            + contract["answer_template"].format(answer_format=formats[family]["answer_format"])
            + contract["joiner"] + contract["missing_template"].format(field_order=", ".join(orders[family])))


def one_line(completion):
    """Outer whitespace follows the unchanged strict scorer's normalization."""
    text = completion.strip()
    return bool(text) and "\n" not in text and "\r" not in text


def error_kind(scored, line_valid):
    if not line_valid:
        return "line_contract_error"
    if not scored["format_valid"]:
        return "format_error"
    if not scored["action_correct"]:
        return "action_error"
    if not scored["decision_correct"]:
        return "decision_error"
    if not scored["value_correct"]:
        return "value_error"
    return "correct"


def _parent(spec, tokenizer):
    parent_spec_path = resolved(spec["parent_spec_path"])
    parent_data_path = resolved(spec["parent_data_path"])
    require(digest(parent_spec_path) == spec["parent_spec_sha256"]
            and digest(parent_data_path) == spec["parent_data_sha256"], "Preserved parent spec/data bytes changed")
    parent = read(parent_spec_path)
    require(parent["schema_version"] == "replica.infrastructure_preflight.v2"
            and resolved(parent["data_path"]) == parent_data_path
            and parent["data_sha256"] == spec["parent_data_sha256"], "Parent data specification differs")
    for key in ("selected_cases", "release_path", "release_manifest_sha256", "source_manifest_sha256",
                "max_new_tokens", "max_length", "seed", "decoding", "model_path", "model_lock"):
        require(spec[key] == parent[key], f"Response contract changes parent selection or fixed setting: {key}")
    require(spec["selected_cases"] == parent["selected_cases"], "Selected case order differs from preserved parent")
    BASE.check_pins(parent["implementation_sha256"], required=BASE.INDEPENDENT_PINS)
    pins = spec["implementation_sha256"]
    require(all(pins.get(name) == value for name, value in parent["implementation_sha256"].items()),
            "Response contract changes preserved parent implementation")
    prior = spec["preserved_baseline"]
    for name in ("generation_report", "generations", "scoring_report"):
        require(digest(resolved(prior[name + "_path"])) == prior[name + "_sha256"],
                f"Preserved baseline {name} bytes changed")
    if "guard_path" in prior:
        require(digest(resolved(prior["guard_path"])) == prior["guard_sha256"],
                "Preserved baseline guard bytes changed")
    old = BASE.score_generation(spec["release_path"], parent_spec_path,
        prior["generation_report_path"], prior["generations_path"], tokenizer=tokenizer)
    archived = read(resolved(prior["scoring_report_path"]))
    require(archived["schema_version"] == BASE.VERSION and archived["examples"] == old["examples"]
            and archived["overall"] == old["overall"], "Preserved baseline score disagrees with independent replay")
    require(old["overall"]["task_correct"]["count"] == 0
            and old["overall"]["hit_token_budget"]["count"] == 16
            and old["overall"]["ended_by_eos"]["count"] == 0,
            "Parent is not the preserved all-capped zero-score baseline")
    return parent, BASE.unique(lines(parent_data_path), "parent data"), old


def score_generation(release, spec_path, report_path, generations_path, *, guard_path=None, tokenizer=None):
    release, spec_path = resolved(release), resolved(spec_path)
    report_path, generations_path = resolved(report_path), resolved(generations_path)
    spec, report = read(spec_path), read(report_path)
    require(spec["schema_version"] == "replica.response_contract_preflight.v1", "Unsupported response-contract specification")
    require(resolved(spec["release_path"]) == release, "Specification names a different release")
    require(digest(release / "manifest.json") == spec["release_manifest_sha256"]
            and digest(release / "source_manifest.json") == spec["source_manifest_sha256"], "Release/source manifest pin differs")
    manifest = read(release / "manifest.json")
    require(manifest["schema_version"] == "replica.infrastructure_slice.v2" and manifest["seed"] == 41
            and manifest["canonical_target_count"] == 2000 and manifest["world_count"] == 1000,
            "Release is not the fixed infrastructure slice")
    for name, expected in manifest["files_sha256"].items():
        require(digest(safe_file(release, name)) == expected, f"Release file bytes changed: {name}")
    BASE.check_pins(spec["implementation_sha256"], required=INDEPENDENT_PINS)
    backend = report["implementation_sha256"]
    require(set(backend) == BASE.BACKEND_PINS and all(spec["implementation_sha256"].get(name) == value
            for name, value in backend.items()), "Native backend implementation differs from specification")
    contract_path = resolved(spec["response_contract_path"])
    require(digest(contract_path) == spec["response_contract_sha256"], "Response contract bytes changed")
    contract = read(contract_path)
    require(spec["prompt_projection"] == contract["projection"]
            and resolved(contract["comparison"]["parent_spec_path"]) == resolved(spec["parent_spec_path"])
            and contract["comparison"]["parent_spec_sha256"] == spec["parent_spec_sha256"]
            and contract["scoring"]["unchanged"] is True
            and digest(resolved(contract["scoring"]["module"])) == contract["scoring"]["sha256"],
            "Projection/parent/unchanged scorer declaration differs")
    comparison = contract["comparison"]
    require(all(comparison[key] == expected for key, expected in {
        "max_new_tokens": 32, "max_length": 512, "seed": 17, "do_sample": False,
        "num_beams": 1, "enable_thinking": False, "device": "mps", "dtype": "float32",
        "cpu_fallback": False, "native_jobs": 1}.items()), "Declared comparison changes fixed native settings")
    preserved = spec["preserved_failed_files_sha256"]
    require(isinstance(preserved, dict) and bool(preserved), "Missing complete preserved failed-file inventory")
    for name, expected in preserved.items():
        require(digest(safe_file(ROOT, name)) == expected, f"Preserved failed file bytes changed: {name}")
    require(digest(resolved(spec["token_accounting_path"])) == spec["token_accounting_sha256"],
            "Predeclared token accounting bytes changed")
    data_path = resolved(spec["data_path"])
    require(digest(data_path) == spec["data_sha256"], "Selected data bytes changed")
    metadata_spec = copy.deepcopy(spec)
    metadata_spec["schema_version"] = "replica.infrastructure_preflight.v2"
    BASE._metadata(metadata_spec, report, data_path)
    model_path, lock_path, lock, config, small_hashes = BASE._source_lock(spec, report)
    require(digest(lock_path) == spec["model_lock_sha256"], "Declared source lock bytes changed")
    tokenizer = BASE.load_tokenizer(model_path) if tokenizer is None else tokenizer
    template_hash = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    require(report["effective_chat_template_sha256"] == template_hash
            and spec["effective_chat_template_sha256"] == template_hash, "Effective chat template differs")
    parent, parent_rows, old = _parent(spec, tokenizer)
    selected = BASE.unique(spec["selected_cases"], "specification")
    rows_list, reports_list, raw_list = lines(data_path), report["examples"], lines(generations_path)
    rows = BASE.unique(rows_list, "selected data")
    reports = BASE.unique(reports_list, "generation report")
    raw = BASE.unique(raw_list, "generation JSONL")
    ids = list(selected)
    require(len(ids) == 16 and all([value["id"] for value in inventory] == ids
            for inventory in (rows_list, reports_list, raw_list)),
            "Selected/report/JSONL IDs must cover the same sixteen cases in fixed order")
    canonical = BASE.unique(lines(release / "canonical/dev_episodes.jsonl"), "canonical dev")
    require(set(ids).issubset(canonical) and len({value["world_id"] for value in selected.values()}) == 16,
            "Selected cases must be sixteen independent canonical dev worlds")
    source_values = read(release / "source_manifest.json")
    sources = {value["source_id"]: value for value in source_values}
    require(len(sources) == len(source_values), "Duplicate source manifest identities")
    eos = report["generation_config"]["eos_token_id"]
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    examples = []
    old_examples = BASE.unique(old["examples"], "independently rescored baseline")
    for case in spec["selected_cases"]:
        identity = case["id"]
        episode, row = canonical[identity], rows[identity]
        require(episode["split"] == "dev" and len(episode["sources"]) == 1, "Selected canonical role/source differs")
        source = sources[episode["sources"][0]["source_id"]]
        expected = solve_episode(episode, release, source)
        require(expected == episode["target"] and all(case[key] == source[key] for key in ("world_id", "family", "scenario"))
                and case["expected_action"] == expected["action"] and case["expected_response"] == expected["response"],
                "Selected expectation differs from independent raw-source truth")
        canonical_row = {"id": identity, "group_id": episode["group_id"], "split": "dev",
                         **input_view(episode, purpose="local_evaluation"),
                         "response": expected["response"], "verified": True}
        require(parent_rows[identity] == canonical_row, "Parent row differs from independently verified canonical export")
        require(hashlib.sha256(canonical_json(canonical_row).encode()).hexdigest()
                == spec["original_case_sha256"][identity], "Original canonical case hash differs")
        projected = {**canonical_row, "prompt": canonical_row["prompt"] + "\n\n" + independent_instruction(source["family"], contract)}
        require(row == projected, "Response projection changes more than the declared static instruction")
        example = reports[identity]
        require(example == raw[identity], "Generation JSONL differs from report example")
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
        prompt = BASE.prompt_ids(row, tokenizer, 512)
        require(example["prompt_tokens"] == len(prompt) and example["prompt_token_ids_sha256"]
                == hashlib.sha256(json.dumps([prompt]).encode()).hexdigest(), "Generated prompt hash/count differs")
        require(type(example["elapsed_seconds"]) in (int, float) and math.isfinite(example["elapsed_seconds"])
                and example["elapsed_seconds"] >= 0, "Invalid generation timing")
        scored = score_completion(example["completion"], expected, family=source["family"],
                                  ended_by_eos=ended, hit_token_budget=capped)
        line_valid = one_line(example["completion"])
        previous = old_examples[identity]
        examples.append({"id": identity, "world_id": source["world_id"], "family": source["family"],
            "scenario": source["scenario"], "expected_action": expected["action"],
            "expected_response": expected["response"], "support_event_ids": expected["support_event_ids"],
            **{key: example[key] for key in ("group_id", "source_ids", "prompt_tokens", "prompt_token_ids_sha256",
                "generated_token_ids", "generated_tokens", "raw_completion", "completion", "elapsed_seconds")},
            **scored, "single_line_compliant": line_valid,
            "contract_valid": line_valid and scored["format_valid"],
            "contract_task_correct": line_valid and scored["task_correct"],
            "error_kind": error_kind(scored, line_valid),
            "paired_previous": {key: previous[key] for key in ("completion", "format_valid", "task_correct",
                "ended_by_eos", "hit_token_budget", "generated_tokens")}})
    guard = BASE._guard(report_path.parent, resolved(guard_path), spec) if guard_path is not None else None
    paired = {"previous_overall": old["overall"], "current_overall": BASE.summary(examples),
        "previous_by_family": old["by_family"], "current_by_family": BASE.grouped(examples, "family"),
        "format_recovered_cases": [value["id"] for value in examples if value["format_valid"]
                                    and not value["paired_previous"]["format_valid"]],
        "format_recovered_but_semantically_wrong": [value["id"] for value in examples
            if value["format_valid"] and not value["task_correct"] and not value["paired_previous"]["format_valid"]],
        "task_improved_cases": [value["id"] for value in examples if value["task_correct"]
                                and not value["paired_previous"]["task_correct"]]}
    return {"schema_version": VERSION, "status": "completed_independent_response_contract_scoring",
        "scope": "Paired descriptive response-contract check on sixteen exposed synthetic dev worlds; no training.",
        "solver_version": SOLVER_VERSION, "scoring_version": SCORING_VERSION,
        "inputs_sha256": {"specification": digest(spec_path), "release_manifest": digest(release / "manifest.json"),
            "source_manifest": digest(release / "source_manifest.json"), "data": digest(data_path),
            "generation_report": digest(report_path), "generations": digest(generations_path),
            "model_lock": digest(lock_path), "response_contract": digest(contract_path),
            "parent_specification": spec["parent_spec_sha256"], "parent_data": spec["parent_data_sha256"],
            "preserved_baseline": spec["preserved_baseline"]},
        "implementation_sha256": spec["implementation_sha256"],
        "source_model": {"repo_id": lock["repo_id"], "revision": lock["revision"], "path": str(model_path),
            "small_files_sha256": small_hashes, "native_weights_rehashed": False},
        "template_sha256": template_hash, "guard_replay": guard,
        "guard_status": "exact_replay_passed" if guard else "not_supplied",
        "overall": BASE.summary(examples), "by_family": BASE.grouped(examples, "family"),
        "by_action": BASE.grouped(examples, "expected_action"), "by_scenario": BASE.grouped(examples, "scenario"),
        "single_line_compliant": {"count": sum(value["single_line_compliant"] for value in examples), "denominator": len(examples)},
        "contract_valid": {"count": sum(value["contract_valid"] for value in examples), "denominator": len(examples)},
        "contract_task_correct": {"count": sum(value["contract_task_correct"] for value in examples), "denominator": len(examples)},
        "format_gate_passed": all(value["contract_valid"] and value["ended_by_eos"]
                                  and not value["hit_token_budget"] for value in examples),
        "error_histogram": dict(Counter(value["error_kind"] for value in examples)),
        "response_histogram": dict(Counter(value["completion"] for value in examples)),
        "paired_comparison": paired, "examples": examples,
        "legacy_fields_ignored": ["XOR-only descriptions", "fixture_task_accuracy", "backend correctness scores"],
        "limitations": ["Instruction wording was chosen after observing the original failed development run",
            "Strict whole-response grammar is unchanged; no answer extraction or semantic grading was added",
            "Single-line metric ignores outer whitespace consistently with the unchanged strict scorer",
            "Same selected worlds, source evidence, targets, tokenizer, model and native budgets; only instruction differs",
            "Sixteen exposed dev cases cannot establish architecture gain, calibration, retention or promotion",
            "Native weights remain covered by launch seal and original retained source evidence"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "spec", "generation-report", "generations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--guard", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Use a fresh output file to preserve earlier scoring evidence")
    result = score_generation(args.release, args.spec, args.generation_report, args.generations, guard_path=args.guard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "examples": result["overall"]["examples"],
        "task_correct": result["overall"]["task_correct"]["count"], "errors": result["error_histogram"]}))


if __name__ == "__main__":
    main()
