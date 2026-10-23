#!/usr/bin/env python3
"""Freeze one public missing-response policy per query before native generation.

Only the existing local tokenizer is used. Original and v1 failed artifacts are
read and hash-bound; neither a model nor an optimizer can be instantiated here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_infrastructure_slice import validate_release
from prepare_response_contract import (
    _parent as original_parent, check_pins, digest, lines, read, relative, require,
    resolved, token_lengths, unique, write,
)
from replica_lab.contracts import canonical_json
from replica_lab.response_contract import load_contract as load_v1_contract, project_row as project_v1_row
from replica_lab.response_policy_contract import instruction_appendix, load_contract, parse_query_policy, project_row

VERSION = "replica.response_policy_preflight.v2"
NEW_PINS = {
    "scripts/prepare_response_policy_contract.py", "src/replica_lab/response_policy_contract.py",
    "scripts/score_response_policy_generations.py",
}
V1_FAILED_FILES = (
    "native_B1_scoring.json", "native_archive_manifest.json",
    "native_B1_raw/generation_report.json", "native_B1_raw/generations.jsonl",
    "native_B1_raw/profile_summary.json", "native_B1_raw/profile_samples.jsonl",
    "native_B1_raw/minicpm5_response_contract_v1_B1_dev16_20261022_guard.json",
    "native_B1_raw/minicpm5_response_contract_v1_B1_dev16_20261022_guard.child.log",
    "native_B1_raw/minicpm5_response_contract_v1_B1_dev16_20261022_guard.host.jsonl",
)


def baseline_paths(directory, guard_name):
    """Hash the exact raw, independently scored and guarded baseline evidence."""
    result = {}
    paths = {"generation_report": directory / "native_B1_raw/generation_report.json",
             "generations": directory / "native_B1_raw/generations.jsonl",
             "scoring_report": directory / "native_B1_scoring.json",
             "guard": directory / "native_B1_raw" / guard_name}
    for name, path in paths.items():
        result[name + "_path"] = relative(path)
        result[name + "_sha256"] = digest(path)
    return result


def _previous_v1(previous_spec_path, parent_spec_path, parent, original_rows, original_failed):
    """Bind the completed v1 failure and prove it changed only its own suffix."""
    previous = read(previous_spec_path)
    require(previous["schema_version"] == "replica.response_contract_preflight.v1",
            "Unsupported previous v1 specification")
    require(resolved(previous["parent_spec_path"]) == parent_spec_path
            and previous["parent_spec_sha256"] == digest(parent_spec_path)
            and previous["parent_data_sha256"] == parent["data_sha256"]
            and resolved(previous["parent_data_path"]) == resolved(parent["data_path"]),
            "Previous v1 specification names a different original parent")
    for key in ("selected_cases", "release_path", "release_manifest_sha256", "source_manifest_sha256",
                "max_new_tokens", "max_length", "seed", "decoding", "role", "model_path", "model_lock",
                "model_lock_sha256", "device", "dtype", "cpu_fallback"):
        require(previous[key] == parent[key], f"Previous v1 changes fixed parent setting: {key}")
    check_pins(previous["implementation_sha256"])
    require(all(previous["implementation_sha256"].get(name) == value
                for name, value in parent["implementation_sha256"].items()),
            "Previous v1 changes preserved original implementation")
    previous_data = resolved(previous["data_path"])
    require(digest(previous_data) == previous["data_sha256"], "Previous v1 data bytes changed")
    previous_rows = unique(lines(previous_data), "previous v1 data")
    require(list(previous_rows) == list(original_rows), "Previous v1 case order differs")
    v1_contract_path = resolved(previous["response_contract_path"])
    require(digest(v1_contract_path) == previous["response_contract_sha256"], "Previous v1 contract bytes changed")
    v1_contract = load_v1_contract(v1_contract_path)
    for case in parent["selected_cases"]:
        identity = case["id"]
        require(previous_rows[identity] == project_v1_row(original_rows[identity], family=case["family"],
                                                       contract=v1_contract),
                "Previous v1 row changes more than its declared static instruction")
        require(previous["original_case_sha256"][identity]
                == hashlib.sha256(canonical_json(original_rows[identity]).encode()).hexdigest(),
                "Previous v1 original case hash differs")
    require(previous["preserved_failed_files_sha256"] == original_failed,
            "Previous v1 does not preserve the complete original failure")
    for name, expected in original_failed.items():
        require(digest(resolved(name)) == expected, "Preserved original failed bytes changed")
    require(digest(resolved(previous["token_accounting_path"])) == previous["token_accounting_sha256"],
            "Previous v1 token accounting bytes changed")
    failed = {}
    for name in V1_FAILED_FILES:
        path = previous_spec_path.parent / name
        require(path.is_file() and not path.is_symlink(), f"Missing previous v1 failed evidence: {name}")
        failed[relative(path)] = digest(path)
    archive = read(previous_spec_path.parent / "native_archive_manifest.json")
    archived_paths = set()
    for record in archive["files"]:
        path = resolved(record["archive_path"])
        require(path.is_relative_to(previous_spec_path.parent) and not path.is_symlink()
                and digest(path) == record["sha256"] and path.stat().st_size == record["bytes"],
                "Previous v1 archive differs from its receipt")
        require(path not in archived_paths, "Duplicate previous v1 archive file")
        archived_paths.add(path)
    require({previous_spec_path.parent / name for name in V1_FAILED_FILES[2:]}.issubset(archived_paths),
            "Previous v1 receipt omits failed native evidence")
    baseline = baseline_paths(previous_spec_path.parent,
        "minicpm5_response_contract_v1_B1_dev16_20261022_guard.json")
    scored = read(resolved(baseline["scoring_report_path"]))
    require(scored["schema_version"] == "replica.response_contract_generation_score.v1"
            and scored["status"] == "completed_independent_response_contract_scoring"
            and scored["inputs_sha256"]["specification"] == digest(previous_spec_path)
            and scored["inputs_sha256"]["data"] == digest(previous_data)
            and scored["inputs_sha256"]["generation_report"] == baseline["generation_report_sha256"]
            and scored["inputs_sha256"]["generations"] == baseline["generations_sha256"],
            "Previous v1 scoring does not bind its original preflight")
    require(scored["overall"]["task_correct"]["count"] == 0
            and scored["overall"]["format_valid"]["count"] == 1
            and scored["overall"]["ended_by_eos"]["count"] == 16
            and scored["overall"]["hit_token_budget"]["count"] == 0,
            "Previous v1 is not the preserved terminated zero-score baseline")
    return previous, previous_data, previous_rows, failed, baseline


def prepare(release, parent_spec, previous_v1_spec, contract_path, output, tokenizer, model_path, model_lock):
    release, parent_spec, previous_v1_spec, contract_path, output = map(resolved,
        (release, parent_spec, previous_v1_spec, contract_path, output))
    model_path, model_lock = map(resolved, (model_path, model_lock))
    require(not output.exists() and not output.is_symlink(), "Use a fresh response-policy directory")
    validate_release(release)
    contract = load_contract(contract_path)
    parent, parent_data, original_rows, original_failed = original_parent(release, parent_spec, model_path, model_lock)
    previous, previous_data, previous_rows, previous_failed, previous_baseline = _previous_v1(
        previous_v1_spec, parent_spec, parent, original_rows, original_failed)
    comparison = contract["comparison"]
    require(resolved(comparison["parent_spec_path"]) == parent_spec
            and comparison["parent_spec_sha256"] == digest(parent_spec)
            and resolved(comparison["previous_v1_spec_path"]) == previous_v1_spec
            and comparison["previous_v1_spec_sha256"] == digest(previous_v1_spec),
            "Policy contract names different preserved original/v1 specifications")
    require(digest(ROOT / contract["scoring"]["module"]) == contract["scoring"]["sha256"]
            and contract["scoring"]["unchanged"] is True, "Policy contract changes the frozen strict parser")
    rubric_path = resolved(contract["scoring"]["judge_rubric_path"])
    rubric = read(rubric_path)
    require(rubric["schema_version"] == "replica.response_semantic_judge_rubric.v1"
            and rubric["status"] == "predeclared_before_native_generation"
            and type(rubric["case_order_seed"]) is int and rubric["case_order_seed"] == 311,
            "Semantic judge rubric is not the frozen predeclared review")
    template_sha256 = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    original_report = read(parent_spec.parent / "native_B1_raw/generation_report.json")
    previous_report = read(previous_v1_spec.parent / "native_B1_raw/generation_report.json")
    require(template_sha256 == original_report["effective_chat_template_sha256"]
            == previous_report["effective_chat_template_sha256"] == previous["effective_chat_template_sha256"],
            "Local tokenizer template differs from preserved baselines")
    new_rows, accounting, policies, original_hashes = [], [], {}, {}
    for case in parent["selected_cases"]:
        identity, family = case["id"], case["family"]
        old, v1 = original_rows[identity], previous_rows[identity]
        policy = parse_query_policy(family, old["prompt"])
        new = project_row(old, family=family, contract=contract)
        lengths = {name: token_lengths(row, tokenizer) for name, row in (("original", old), ("v1", v1), ("v2", new))}
        require(all(value["native_full"] <= 512 and value["native_prompt"] <= 512
                    and value["source_writer"] <= 512 for value in lengths.values()),
                f"Response-policy row {identity} exceeds fixed512-token admission; no truncation")
        require(len({value["supervised_answer_plus_eos"] for value in lengths.values()}) == 1
                and len({value["source_writer"] for value in lengths.values()}) == 1,
                "Projection changes supervision/evidence token lengths")
        policies[identity] = policy
        original_hashes[identity] = hashlib.sha256(canonical_json(old).encode()).hexdigest()
        appendix = instruction_appendix(family, old["prompt"], contract)
        new_rows.append(new)
        accounting.append({"id": identity, "family": family, "query_policy": policy, **lengths,
            "appendix_in_isolation_tokens": len(tokenizer.encode(contract["append_separator"] + appendix,
                                                                  add_special_tokens=False)),
            "v2_minus_original_native_prompt_tokens": lengths["v2"]["native_prompt"] - lengths["original"]["native_prompt"],
            "v2_minus_v1_native_prompt_tokens": lengths["v2"]["native_prompt"] - lengths["v1"]["native_prompt"]})
    require(len(new_rows) == 16 and [row["id"] for row in new_rows]
            == [case["id"] for case in parent["selected_cases"]], "Projected cases/order differ from original")
    pins = dict(previous["implementation_sha256"])
    for name in sorted(NEW_PINS):
        require((ROOT / name).is_file(), f"Freeze response-policy implementation before preparation: {name}")
        pins[name] = digest(ROOT / name)
    pins[relative(contract_path)] = digest(contract_path)
    pins[relative(rubric_path)] = digest(rubric_path)
    output.mkdir(parents=True)
    data_path, accounting_path = output / "dev_cases.jsonl", output / "token_accounting.json"
    with data_path.open("x", encoding="utf-8") as stream:
        stream.write("".join(canonical_json(row) + "\n" for row in new_rows))
    write(accounting_path, {"schema_version": "replica.response-policy-token-accounting.v2",
        "examples": 16, "max_length": 512, "truncated_examples": 0, "overlong_ids": [],
        "tokenizer_path": relative(model_path), "model_lock_sha256": digest(model_lock),
        "response_contract_sha256": digest(contract_path), "parent_data_sha256": digest(parent_data),
        "previous_v1_data_sha256": digest(previous_data),
        **{name + "_prompt_tokens": sum(record[name]["native_prompt"] for record in accounting)
           for name in ("original", "v1", "v2")},
        **{name + "_max_full_tokens": max(record[name]["native_full"] for record in accounting)
           for name in ("original", "v1", "v2")},
        "scope": "Local tokenizer only; no model instantiated. All original/v1/v2 prompt, full-reference and source-writer streams admitted without truncation. Supervision, EOS and evidence remain unchanged.",
        "examples_detail": accounting})
    spec = {**parent, "schema_version": VERSION, "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_spec_path": relative(parent_spec), "parent_spec_sha256": digest(parent_spec),
        "parent_data_path": relative(parent_data), "parent_data_sha256": digest(parent_data),
        "previous_v1_spec_path": relative(previous_v1_spec), "previous_v1_spec_sha256": digest(previous_v1_spec),
        "previous_v1_data_path": relative(previous_data), "previous_v1_data_sha256": digest(previous_data),
        "previous_v1_baseline": previous_baseline,
        "response_contract_path": relative(contract_path), "response_contract_sha256": digest(contract_path),
        "judge_rubric_path": relative(rubric_path), "judge_rubric_sha256": digest(rubric_path),
        "judge_case_order_seed": rubric["case_order_seed"],
        "prompt_projection": contract["projection"], "query_policy_by_id": policies,
        "data_path": relative(data_path), "data_sha256": digest(data_path),
        "original_case_sha256": original_hashes,
        "preserved_failed_files_sha256": {**original_failed, **previous_failed},
        "preserved_baseline": baseline_paths(parent_spec.parent,
            "minicpm5_infrastructure_v2_B1_dev16_20261022_guard.json"),
        "token_accounting_path": relative(accounting_path), "token_accounting_sha256": digest(accounting_path),
        "effective_chat_template_sha256": template_sha256, "implementation_sha256": pins,
        "generation_command": [".venv/bin/python", "scripts/run_generation_preflight.py", "--report",
            "runs/minicpm5_response_policy_v2_B1_dev16_20261022_guard.json", "--", "--model-path", relative(model_path),
            "--model-lock", relative(model_lock), "--data", relative(data_path), "--split", "dev", "--output",
            "runs/minicpm5_response_policy_v2_B1_dev16_20261022", "--device", "mps", "--max-examples", "16",
            "--max-new-tokens", "32", "--max-length", "512", "--seed", "17"],
        "scoring_command": [".venv/bin/python", "scripts/score_response_policy_generations.py", "--release", relative(release),
            "--spec", relative(output / "evaluation_spec.json"), "--generation-report",
            "runs/minicpm5_response_policy_v2_B1_dev16_20261022/generation_report.json", "--generations",
            "runs/minicpm5_response_policy_v2_B1_dev16_20261022/generations.jsonl", "--guard",
            "runs/minicpm5_response_policy_v2_B1_dev16_20261022_guard.json", "--output", relative(output / "native_B1_scoring.json")],
        "scoring_settings": {"grammar_module": contract["scoring"]["module"],
            "grammar_sha256": contract["scoring"]["sha256"], "unchanged": True,
            "single_line_compliance": contract["scoring"]["single_line_compliance"],
            "format_gate": contract["scoring"]["format_gate"], "task_correctness_separate": True},
        "scope": "One predeclared public-policy prompt follow-up to two preserved failed original-model B1 runs. Same sixteen exposed development worlds, evidence, labels, model, template and native budgets. No training, promotion or independent confirmation.",
        "unchanged_conditions": ["ordered case IDs", "source bytes and eligible evidence", "reference answers", "split roles",
            "source model and local tokenizer", "native chat template", "max_new_tokens32", "max_length512", "seed17",
            "greedy decoding and thinking disabled", "MPS FP32 with CPU fallback disabled", "strict whole-response parser"],
        "response_contract_delta": "Append one deterministic family instruction plus exactly the missing-response policy explicitly requested by the canonical public query. No target/action/value/evidence/missingness chooses the instruction."}
    write(output / "evaluation_spec.json", spec)
    return {"examples": 16, "output": relative(output), "data_sha256": spec["data_sha256"],
            "max_full_tokens": max(record["v2"]["native_full"] for record in accounting),
            "preserved_failed_files": len(spec["preserved_failed_files_sha256"]),
            "parent_spec_sha256": spec["parent_spec_sha256"], "previous_v1_spec_sha256": spec["previous_v1_spec_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "parent-spec", "previous-v1-spec", "contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=ROOT / "checkpoints/minicpm5_source")
    parser.add_argument("--model-lock", type=Path, default=ROOT / "configs/minicpm5_sealed_lock.local.json")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    import json
    print(json.dumps(prepare(args.release, args.parent_spec, args.previous_v1_spec, args.contract, args.output,
                             tokenizer, args.model_path, args.model_lock)))


if __name__ == "__main__":
    main()
