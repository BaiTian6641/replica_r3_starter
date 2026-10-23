#!/usr/bin/env python3
"""Recheck a public-query-policy projection against both preserved failed runs.

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
import re
from string import Formatter
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("response_policy_previous_scorer", ROOT / "scripts/score_response_contract_generations.py")
V1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V1)
BASE = V1.BASE

from replica_lab.episodes import input_view
from replica_lab.contracts import canonical_json
from replica_lab.infrastructure_scoring import SCORING_VERSION, score_completion
from replica_lab.infrastructure_solver import SOLVER_VERSION, solve_episode

VERSION = "replica.response_policy_generation_score.v2"
INDEPENDENT_PINS = V1.INDEPENDENT_PINS | {
    "scripts/score_response_policy_generations.py", "src/replica_lab/response_policy_contract.py",
    "scripts/prepare_response_policy_contract.py", "configs/response_policy_contract_v2.json",
    "configs/response_policy_judge_rubric_v1.json"}
require, digest, read, lines, resolved, safe_file = (
    BASE.require, BASE.digest, BASE.read, BASE.lines, BASE.resolved, BASE.safe_file)


def independent_instruction(family, query, contract):
    """Reconstruct the appendix using only public query syntax and static data.

    This does not call the projection module or inspect source records, labels,
    target actions, or whether any required facts are actually missing.
    """
    require(contract["schema_version"] == "replica.response_policy_contract.v2"
            and contract["status"] == "predeclared_before_native_generation"
            and contract["append_separator"] == "\n\n" and contract["joiner"] == "\n"
            and contract["projection"] == "canonical_prompt_plus_separator_plus_public_query_policy_instruction_v2"
            and contract["projection_inputs"] == ["family", "canonical_query", "static_contract_only"],
            "Unsupported public-policy response contract")
    orders = {"register_xor8": ["RULE", "INPUT", "MASK"], "affine_mod97": ["RULE", "A", "X", "B"],
              "finite_state": ["RULE", "START", "SYMBOLS"], "tool_recovery": ["RULE", "STATUS", "ATTEMPT", "LIMIT"]}
    bodies = {
        "register_xor8": "compute INPUT XOR MASK using the observed rule",
        "affine_mod97": "compute the affine result using the observed rule",
        "finite_state": "report the final state after consuming SYMBOLS from START using the observed transition rule",
        "tool_recovery": "recommend the next operation for fictional tool lookup using the observed retry rule"}
    answer_formats = {"register_xor8": "only the base-10 integer result.",
        "affine_mod97": "only the base-10 integer result.", "finite_state": "only S0, S1, S2, or S3.",
        "tool_recovery": "exactly RETRY lookup, VERIFY receipt, or STOP."}
    require(set(contract["family_formats"]) == set(orders) and family in orders,
            "Contract family inventory differs")
    formats = contract["family_formats"]
    require(all(value == {"field_order": orders[name], "answer_format": answer_formats[name]}
                for name, value in formats.items()), "Contract family format differs from canonical grammar")
    match = re.fullmatch(r"At observation cutoff (?:0|[1-9][0-9]*), " + re.escape(bodies[family])
        + r"\. If a required field is unavailable, (ask for it|respond UNKNOWN)\.", query)
    require(match is not None, "Public query does not match the complete canonical family grammar")
    policy = "ask" if match.group(1) == "ask for it" else "unknown"
    policies = contract["missing_policy_templates"]
    require(isinstance(policies, dict) and set(policies) == {"ask", "unknown"},
            "Contract missing-policy inventory differs")
    for name, field in (("required_items_template", "field_order"), ("answer_template", "answer_format")):
        try:
            parsed = list(Formatter().parse(contract[name]))
        except (ValueError, TypeError) as error:
            raise ValueError("Malformed response-policy template") from error
        require([item[1] for item in parsed if item[1] is not None] == [field]
                and all(not item[2] and item[3] is None for item in parsed),
                "Response-policy template has undeclared projection inputs")
    for value in (contract["common_prefix"], contract["evidence_instruction"],
                  contract["required_items_template"], contract["answer_template"], *policies.values()):
        require(isinstance(value, str) and bool(value.strip()) and "\n" not in value and "\r" not in value,
                "Response-policy instruction parts must be nonempty single lines")
    require(all(not any(item[1] is not None for item in Formatter().parse(value))
                for value in policies.values()), "Missing-policy wording has undeclared projection inputs")
    instruction = contract["joiner"].join((contract["common_prefix"], contract["evidence_instruction"],
        contract["required_items_template"].format(field_order=", ".join(orders[family])),
        contract["answer_template"].format(answer_format=formats[family]["answer_format"]), policies[policy]))
    return policy, instruction


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


def _previous_runs(spec, tokenizer):
    """Replay both archived failures without changing either old scoring implementation."""
    parent, parent_rows, original = V1._parent(spec, tokenizer)
    previous_spec_path = resolved(spec["previous_v1_spec_path"])
    previous_data_path = resolved(spec["previous_v1_data_path"])
    require(digest(previous_spec_path) == spec["previous_v1_spec_sha256"]
            and digest(previous_data_path) == spec["previous_v1_data_sha256"],
            "Preserved v1 specification/data bytes changed")
    previous_spec = read(previous_spec_path)
    require(previous_spec["schema_version"] == "replica.response_contract_preflight.v1"
            and resolved(previous_spec["data_path"]) == previous_data_path
            and previous_spec["data_sha256"] == spec["previous_v1_data_sha256"],
            "Preserved v1 data specification differs")
    for key in ("parent_spec_path", "parent_spec_sha256", "parent_data_path", "parent_data_sha256",
                "selected_cases", "release_path", "release_manifest_sha256", "source_manifest_sha256",
                "max_new_tokens", "max_length", "seed", "decoding", "model_path", "model_lock"):
        require(spec[key] == previous_spec[key], f"Policy contract changes v1 selection or fixed setting: {key}")
    require(all(spec["implementation_sha256"].get(name) == value
                for name, value in previous_spec["implementation_sha256"].items()),
            "Policy contract changes preserved v1 implementation")
    require(spec["preserved_baseline"] == previous_spec["preserved_baseline"],
            "Policy contract changes the preserved original baseline inventory")
    require(all(spec["preserved_failed_files_sha256"].get(name) == value
                for name, value in previous_spec["preserved_failed_files_sha256"].items()),
            "Policy contract omits or changes preserved earlier failed-file pins")
    prior = spec["previous_v1_baseline"]
    for name in ("generation_report", "generations", "scoring_report"):
        require(digest(resolved(prior[name + "_path"])) == prior[name + "_sha256"],
                f"Preserved v1 {name} bytes changed")
    guard = None
    if "guard_path" in prior:
        guard = resolved(prior["guard_path"])
        require(digest(guard) == prior["guard_sha256"], "Preserved v1 guard bytes changed")
    previous = V1.score_generation(spec["release_path"], previous_spec_path,
        prior["generation_report_path"], prior["generations_path"],
        guard_path=guard, tokenizer=tokenizer)
    archived = read(resolved(prior["scoring_report_path"]))
    require(archived["schema_version"] == V1.VERSION
            and all(archived[key] == previous[key] for key in (
                "examples", "overall", "single_line_compliant", "contract_valid", "contract_task_correct",
                "error_histogram", "format_gate_passed")),
            "Preserved v1 score disagrees with independent replay")
    require(previous["overall"]["task_correct"]["count"] == 0
            and previous["overall"]["ended_by_eos"]["count"] == 16
            and previous["overall"]["hit_token_budget"]["count"] == 0
            and previous["single_line_compliant"]["count"] == 16,
            "Previous v1 is not the preserved terminated zero-score baseline")
    original_guard = spec["preserved_baseline"].get("guard_path")
    if original_guard is not None:
        BASE._guard(resolved(spec["preserved_baseline"]["generation_report_path"]).parent,
                    resolved(original_guard), parent)
    return parent, parent_rows, original, previous


def score_generation(release, spec_path, report_path, generations_path, *, guard_path=None, tokenizer=None):
    release, spec_path = resolved(release), resolved(spec_path)
    report_path, generations_path = resolved(report_path), resolved(generations_path)
    spec, report = read(spec_path), read(report_path)
    require(spec["schema_version"] == "replica.response_policy_preflight.v2", "Unsupported response-policy specification")
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
            and resolved(contract["comparison"]["previous_v1_spec_path"]) == resolved(spec["previous_v1_spec_path"])
            and contract["comparison"]["previous_v1_spec_sha256"] == spec["previous_v1_spec_sha256"]
            and contract["scoring"]["unchanged"] is True
            and digest(resolved(contract["scoring"]["module"])) == contract["scoring"]["sha256"],
            "Projection/parent/unchanged scorer declaration differs")
    comparison = contract["comparison"]
    require(all(comparison[key] == expected for key, expected in {
        "max_new_tokens": 32, "max_length": 512, "seed": 17, "do_sample": False,
        "num_beams": 1, "enable_thinking": False, "device": "mps", "dtype": "float32",
        "cpu_fallback": False, "native_jobs": 1}.items()), "Declared comparison changes fixed native settings")
    require(resolved(comparison["model_path"]) == resolved(spec["model_path"])
            and resolved(comparison["model_lock"]) == resolved(spec["model_lock"]),
            "Declared comparison changes the original source model")
    rubric_path = resolved(spec["judge_rubric_path"])
    require(resolved(contract["scoring"]["judge_rubric_path"]) == rubric_path
            and digest(rubric_path) == spec["judge_rubric_sha256"]
            and spec["judge_case_order_seed"] == 311,
            "Predeclared separate judge rubric or case-order seed differs")
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
    parent, parent_rows, original, previous_v1 = _previous_runs(spec, tokenizer)
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
    original_examples = BASE.unique(original["examples"], "independently rescored original baseline")
    previous_examples = BASE.unique(previous_v1["examples"], "independently rescored v1 baseline")
    require(isinstance(spec["query_policy_by_id"], dict) and set(spec["query_policy_by_id"]) == set(ids),
            "Public query-policy inventory differs")
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
        policy, instruction = independent_instruction(source["family"], canonical_row["prompt"], contract)
        require(spec["query_policy_by_id"][identity] == policy, "Declared query policy differs from public canonical query")
        projected = {**canonical_row, "prompt": canonical_row["prompt"] + "\n\n" + instruction}
        require(row == projected, "Response projection changes more than the declared public-policy instruction")
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
        previous, initial = previous_examples[identity], original_examples[identity]
        examples.append({"id": identity, "world_id": source["world_id"], "family": source["family"],
            "scenario": source["scenario"], "query_missing_policy": policy, "expected_action": expected["action"],
            "expected_response": expected["response"], "support_event_ids": expected["support_event_ids"],
            **{key: example[key] for key in ("group_id", "source_ids", "prompt_tokens", "prompt_token_ids_sha256",
                "generated_token_ids", "generated_tokens", "raw_completion", "completion", "elapsed_seconds")},
            **scored, "single_line_compliant": line_valid,
            "contract_valid": line_valid and scored["format_valid"],
            "contract_task_correct": line_valid and scored["task_correct"],
            "error_kind": error_kind(scored, line_valid),
            "paired_previous": {key: previous[key] for key in ("completion", "format_valid", "task_correct",
                "ended_by_eos", "hit_token_budget", "generated_tokens")},
            "paired_original": {key: initial[key] for key in ("completion", "format_valid", "task_correct",
                "ended_by_eos", "hit_token_budget", "generated_tokens")}})
    guard = BASE._guard(report_path.parent, resolved(guard_path), spec) if guard_path is not None else None
    paired = {"original_overall": original["overall"], "previous_v1_overall": previous_v1["overall"],
        "current_overall": BASE.summary(examples), "original_by_family": original["by_family"],
        "previous_v1_by_family": previous_v1["by_family"], "current_by_family": BASE.grouped(examples, "family"),
        "original_by_action": original["by_action"], "previous_v1_by_action": previous_v1["by_action"],
        "current_by_action": BASE.grouped(examples, "expected_action"),
        "format_recovered_cases": [value["id"] for value in examples if value["format_valid"]
                                    and not value["paired_previous"]["format_valid"]],
        "format_recovered_but_semantically_wrong": [value["id"] for value in examples
            if value["format_valid"] and not value["task_correct"] and not value["paired_previous"]["format_valid"]],
        "task_improved_cases": [value["id"] for value in examples if value["task_correct"]
                                and not value["paired_previous"]["task_correct"]]}
    return {"schema_version": VERSION, "status": "completed_independent_response_policy_scoring",
        "scope": "Three-run descriptive public-policy contract check on sixteen exposed synthetic dev worlds; no training.",
        "solver_version": SOLVER_VERSION, "scoring_version": SCORING_VERSION,
        "inputs_sha256": {"specification": digest(spec_path), "release_manifest": digest(release / "manifest.json"),
            "source_manifest": digest(release / "source_manifest.json"), "data": digest(data_path),
            "generation_report": digest(report_path), "generations": digest(generations_path),
            "model_lock": digest(lock_path), "response_contract": digest(contract_path),
            "parent_specification": spec["parent_spec_sha256"], "parent_data": spec["parent_data_sha256"],
            "preserved_baseline": spec["preserved_baseline"],
            "previous_v1_specification": spec["previous_v1_spec_sha256"],
            "previous_v1_data": spec["previous_v1_data_sha256"],
            "previous_v1_baseline": spec["previous_v1_baseline"]},
        "implementation_sha256": spec["implementation_sha256"],
        "source_model": {"repo_id": lock["repo_id"], "revision": lock["revision"], "path": str(model_path),
            "small_files_sha256": small_hashes, "native_weights_rehashed": False},
        "template_sha256": template_hash, "guard_replay": guard,
        "guard_status": "exact_replay_passed" if guard else "not_supplied",
        "overall": BASE.summary(examples), "by_family": BASE.grouped(examples, "family"),
        "by_action": BASE.grouped(examples, "expected_action"), "by_scenario": BASE.grouped(examples, "scenario"),
        "by_query_missing_policy": BASE.grouped(examples, "query_missing_policy"),
        "error_by_family": {family: dict(Counter(value["error_kind"] for value in examples if value["family"] == family))
            for family in sorted({value["family"] for value in examples})},
        "error_by_action": {action: dict(Counter(value["error_kind"] for value in examples if value["expected_action"] == action))
            for action in sorted({value["expected_action"] for value in examples})},
        "single_line_compliant": {"count": sum(value["single_line_compliant"] for value in examples), "denominator": len(examples)},
        "contract_valid": {"count": sum(value["contract_valid"] for value in examples), "denominator": len(examples)},
        "contract_task_correct": {"count": sum(value["contract_task_correct"] for value in examples), "denominator": len(examples)},
        "format_gate_passed": all(value["contract_valid"] and value["ended_by_eos"]
                                  and not value["hit_token_budget"] for value in examples),
        "error_histogram": dict(Counter(value["error_kind"] for value in examples)),
        "response_histogram": dict(Counter(value["completion"] for value in examples)),
        "paired_comparison": paired, "examples": examples,
        "legacy_fields_ignored": ["XOR-only descriptions", "fixture_task_accuracy", "backend correctness scores"],
        "limitations": ["Instruction wording was chosen after observing both failed development runs",
            "Strict whole-response grammar is unchanged; no answer extraction or semantic grading is performed by this scorer",
            "A separate authorized subagent semantic judgment may be reported with its own rubric and rationale",
            "The appendix uses only public family and the query's stated missing-response policy, never true missingness or labels",
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
