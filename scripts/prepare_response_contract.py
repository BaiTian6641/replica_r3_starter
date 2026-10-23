#!/usr/bin/env python3
"""Lock a prompt-only follow-up to the preserved sixteen-case native failure.

Only a local tokenizer is loaded. Model inference belongs to the separately
guarded native backend; this program cannot instantiate or download a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_infrastructure_slice import validate_release
from prepare_infrastructure_preflight import select_cases
from train_text import encode_row
from replica_lab.contracts import canonical_json
from replica_lab.episodes import export_sft, verify_episode
from replica_lab.infrastructure_solver import independent_check, solve_episode
from replica_lab.response_contract import instruction_appendix, load_contract, project_row

VERSION = "replica.response_contract_preflight.v1"
BACKEND_PINS = {
    "scripts/evaluate_text.py", "scripts/train_text.py", "src/replica_lab/model_adapter.py",
    "src/replica_lab/generation.py", "src/replica_lab/neural.py", "src/replica_lab/evaluation.py"}
NEW_PINS = {
    "scripts/prepare_response_contract.py", "src/replica_lab/response_contract.py",
    "scripts/score_response_contract_generations.py"}
FAILED_FILES = (
    "native_B1_scoring.json", "native_archive_manifest.json",
    "native_B1_raw/generation_report.json", "native_B1_raw/generations.jsonl",
    "native_B1_raw/profile_summary.json", "native_B1_raw/profile_samples.jsonl",
    "native_B1_raw/minicpm5_infrastructure_v2_B1_dev16_20261022_guard.json",
    "native_B1_raw/minicpm5_infrastructure_v2_B1_dev16_20261022_guard.child.log",
    "native_B1_raw/minicpm5_infrastructure_v2_B1_dev16_20261022_guard.host.jsonl")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def resolved(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def unique(records, name):
    require(isinstance(records, list) and all(isinstance(record, dict) and isinstance(record.get("id"), str)
                                            for record in records), f"Malformed {name} inventory")
    result = {record["id"]: record for record in records}
    require(len(result) == len(records), f"Duplicate IDs in {name}")
    return result


def check_pins(pins):
    require(isinstance(pins, dict) and BACKEND_PINS.issubset(pins), "Missing parent backend pins")
    for name, expected in pins.items():
        relative_name = Path(name)
        require(not relative_name.is_absolute() and ".." not in relative_name.parts,
                "Parent implementation pin escapes repository")
        path = ROOT / relative_name
        require(not path.is_symlink() and path.resolve().is_relative_to(ROOT)
                and digest(path) == expected, f"Parent implementation bytes changed: {name}")


def token_lengths(row, tokenizer):
    batch = encode_row(row, tokenizer, 16384)
    prompt, full, support = (batch[key].numel() for key in ("prompt_ids", "input_ids", "memory_ids"))
    return {"native_prompt": prompt, "native_full": full, "source_writer": support,
            "supervised_answer_plus_eos": full - prompt, "answer": full - prompt - 1, "eos": 1}


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _parent(release, parent_spec, model_path, model_lock):
    spec = read(parent_spec)
    require(spec["schema_version"] == "replica.infrastructure_preflight.v2", "Unsupported parent specification")
    require((spec["max_new_tokens"], spec["max_length"], spec["seed"], spec["decoding"], spec["role"],
             spec["device"], spec["dtype"], spec["cpu_fallback"])
            == (32, 512, 17, "greedy", "dev", "mps", "float32", False), "Parent fixed settings differ")
    require(resolved(spec["release_path"]) == release, "Parent release differs")
    require(digest(release / "manifest.json") == spec["release_manifest_sha256"]
            and digest(release / "source_manifest.json") == spec["source_manifest_sha256"], "Parent release/source pin differs")
    require(resolved(spec["model_path"]) == model_path and resolved(spec["model_lock"]) == model_lock
            and digest(model_lock) == spec["model_lock_sha256"], "Parent local model/lock differs")
    require(model_path == (ROOT / "checkpoints/minicpm5_source").resolve()
            and model_lock == (ROOT / "configs/minicpm5_sealed_lock.local.json").resolve(),
            "Only the sealed original local source is admitted")
    check_pins(spec["implementation_sha256"])
    data_path = resolved(spec["data_path"])
    require(digest(data_path) == spec["data_sha256"], "Parent selected data bytes changed")
    selected = unique(spec["selected_cases"], "parent specification")
    rows_list = lines(data_path)
    rows = unique(rows_list, "parent data")
    require(len(selected) == 16 and list(selected) == list(rows)
            and len({case["world_id"] for case in selected.values()}) == 16,
            "Parent must preserve sixteen ordered distinct development worlds")
    sources_list = read(release / "source_manifest.json")
    sources = {record["source_id"]: record for record in sources_list}
    require(len(sources) == len(sources_list), "Duplicate source manifest IDs")
    payloads = lines(release / "canonical/dev_episodes.jsonl")
    canonical = unique(payloads, "canonical development")
    require([case["id"] for case in select_cases(payloads, sources)] == list(selected),
            "Parent case order/selection differs from the fixed original rule")
    require(Counter(case["expected_action"] for case in selected.values()) == {"answer": 8, "ask": 4, "unknown": 4},
            "Parent case action balance differs")
    solver_hash = digest(ROOT / "src/replica_lab/infrastructure_solver.py")
    for identity, case in selected.items():
        episode = canonical[identity]
        require(episode["split"] == "dev" and len(episode["sources"]) == 1, "Parent case is not single-source development")
        source = sources[episode["sources"][0]["source_id"]]
        expected = solve_episode(episode, release, source)
        require(expected == episode["target"] and case["world_id"] == episode["group_id"]
                and all(case[key] == source[key] for key in ("world_id", "family", "scenario"))
                and case["expected_action"] == expected["action"] and case["expected_response"] == expected["response"],
                "Parent expectation differs from independent raw-source truth")
        receipt = verify_episode(episode, verifier_id="replica.infrastructure.raw-source-solver.v2",
            verifier_sha256=solver_hash, verifier=lambda value, s=source: independent_check(value, release, s))
        require(receipt.passed and export_sft(episode, purpose="local_evaluation", verification=receipt) == rows[identity],
                "Parent row differs from independently verified canonical export")
    failed = {}
    for name in FAILED_FILES:
        path = parent_spec.parent / name
        require(path.is_file() and not path.is_symlink(), f"Missing preserved failed evidence: {name}")
        failed[relative(path)] = digest(path)
    archive = read(parent_spec.parent / "native_archive_manifest.json")
    archived_paths = set()
    for record in archive["files"]:
        path = resolved(record["archive_path"])
        require(path.is_relative_to(parent_spec.parent) and not path.is_symlink()
                and digest(path) == record["sha256"] and path.stat().st_size == record["bytes"],
                "Preserved failed archive differs from its original receipt")
        require(path not in archived_paths, "Duplicate preserved archive file")
        archived_paths.add(path)
    require({parent_spec.parent / name for name in FAILED_FILES[2:]}.issubset(archived_paths),
            "Original archive receipt omits required failed evidence")
    scored = read(parent_spec.parent / "native_B1_scoring.json")
    require(scored["schema_version"] == "replica.infrastructure_generation_score.v2"
            and scored["status"] == "completed_independent_infrastructure_scoring"
            and scored["inputs_sha256"]["specification"] == digest(parent_spec)
            and scored["inputs_sha256"]["data"] == digest(data_path)
            and scored["inputs_sha256"]["generation_report"] == digest(parent_spec.parent / "native_B1_raw/generation_report.json")
            and scored["inputs_sha256"]["generations"] == digest(parent_spec.parent / "native_B1_raw/generations.jsonl"),
            "Preserved scoring evidence does not bind the original parent preflight")
    return spec, data_path, rows, failed


def prepare(release, parent_spec, contract_path, output, tokenizer, model_path, model_lock):
    release, parent_spec, contract_path, output = map(resolved, (release, parent_spec, contract_path, output))
    model_path, model_lock = map(resolved, (model_path, model_lock))
    require(not output.exists() and not output.is_symlink(), "Use a fresh response-contract directory")
    validate_release(release)
    contract = load_contract(contract_path)
    parent, parent_data, old_rows, failed = _parent(release, parent_spec, model_path, model_lock)
    require(resolved(contract["comparison"]["parent_spec_path"]) == parent_spec
            and contract["comparison"]["parent_spec_sha256"] == digest(parent_spec),
            "Response contract names a different parent specification")
    require(digest(ROOT / contract["scoring"]["module"]) == contract["scoring"]["sha256"]
            and contract["scoring"]["unchanged"] is True, "Response contract changes the frozen parser")
    template_sha256 = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    baseline_report = read(parent_spec.parent / "native_B1_raw/generation_report.json")
    require(template_sha256 == baseline_report["effective_chat_template_sha256"],
            "Local tokenizer template differs from preserved baseline")
    new_rows, accounting = [], []
    original_case_sha256 = {}
    for case in parent["selected_cases"]:
        identity, family = case["id"], case["family"]
        old = old_rows[identity]
        new = project_row(old, family=family, contract=contract)
        old_length, new_length = token_lengths(old, tokenizer), token_lengths(new, tokenizer)
        appendix = instruction_appendix(family, contract)
        appendix_tokens = len(tokenizer.encode(contract["append_separator"] + appendix, add_special_tokens=False))
        require(new_length["native_full"] <= 512 and new_length["source_writer"] <= 512,
                f"Response-contract row {identity} exceeds fixed512-token admission; no truncation")
        require(old_length["supervised_answer_plus_eos"] == new_length["supervised_answer_plus_eos"]
                and old_length["source_writer"] == new_length["source_writer"], "Projection changes supervision/evidence")
        original_case_sha256[identity] = hashlib.sha256(canonical_json(old).encode()).hexdigest()
        new_rows.append(new)
        accounting.append({"id": identity, "family": family, "old": old_length, "new": new_length,
            "appendix_in_isolation_tokens": appendix_tokens,
            "native_prompt_token_delta": new_length["native_prompt"] - old_length["native_prompt"]})
    require(len(new_rows) == 16 and [row["id"] for row in new_rows] == [case["id"] for case in parent["selected_cases"]],
            "Projected cases/order differ from parent")
    pins = dict(parent["implementation_sha256"])
    for name in sorted(NEW_PINS):
        require((ROOT / name).is_file(), f"Freeze response-contract implementation before preparation: {name}")
        pins[name] = digest(ROOT / name)
    pins[relative(contract_path)] = digest(contract_path)
    output.mkdir(parents=True)
    data_path = output / "dev_cases.jsonl"
    with data_path.open("x", encoding="utf-8") as stream:
        stream.write("".join(canonical_json(row) + "\n" for row in new_rows))
    accounting_path = output / "token_accounting.json"
    write(accounting_path, {"schema_version": "replica.response-contract-token-accounting.v1",
        "examples": 16, "max_length": 512, "truncated_examples": 0, "overlong_ids": [],
        "tokenizer_path": relative(model_path), "model_lock_sha256": digest(model_lock),
        "response_contract_sha256": digest(contract_path), "parent_data_sha256": digest(parent_data),
        "old_prompt_tokens": sum(record["old"]["native_prompt"] for record in accounting),
        "new_prompt_tokens": sum(record["new"]["native_prompt"] for record in accounting),
        "old_max_full_tokens": max(record["old"]["native_full"] for record in accounting),
        "new_max_full_tokens": max(record["new"]["native_full"] for record in accounting),
        "scope": "Local tokenizer only; no model instantiated. Supervised answer/EOS and evidence unchanged. Appendix isolation tokens need not equal rendered prompt token delta.",
        "examples_detail": accounting})
    spec = {**parent, "schema_version": VERSION,
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_spec_path": relative(parent_spec), "parent_spec_sha256": digest(parent_spec),
        "parent_data_path": relative(parent_data), "parent_data_sha256": digest(parent_data),
        "response_contract_path": relative(contract_path), "response_contract_sha256": digest(contract_path),
        "prompt_projection": contract["projection"],
        "data_path": relative(data_path), "data_sha256": digest(data_path),
        "original_case_sha256": original_case_sha256, "preserved_failed_files_sha256": failed,
        "preserved_baseline": {
            "generation_report_path": relative(parent_spec.parent / "native_B1_raw/generation_report.json"),
            "generation_report_sha256": digest(parent_spec.parent / "native_B1_raw/generation_report.json"),
            "generations_path": relative(parent_spec.parent / "native_B1_raw/generations.jsonl"),
            "generations_sha256": digest(parent_spec.parent / "native_B1_raw/generations.jsonl"),
            "scoring_report_path": relative(parent_spec.parent / "native_B1_scoring.json"),
            "scoring_report_sha256": digest(parent_spec.parent / "native_B1_scoring.json"),
            "guard_path": relative(parent_spec.parent / "native_B1_raw/minicpm5_infrastructure_v2_B1_dev16_20261022_guard.json"),
            "guard_sha256": digest(parent_spec.parent / "native_B1_raw/minicpm5_infrastructure_v2_B1_dev16_20261022_guard.json")},
        "token_accounting_path": relative(accounting_path), "token_accounting_sha256": digest(accounting_path),
        "effective_chat_template_sha256": template_sha256,
        "implementation_sha256": pins,
        "generation_command": [".venv/bin/python", "scripts/run_generation_preflight.py", "--report",
            "runs/minicpm5_response_contract_v1_B1_dev16_20261022_guard.json", "--", "--model-path", relative(model_path),
            "--model-lock", relative(model_lock), "--data", relative(data_path), "--split", "dev", "--output",
            "runs/minicpm5_response_contract_v1_B1_dev16_20261022", "--device", "mps", "--max-examples", "16",
            "--max-new-tokens", "32", "--max-length", "512", "--seed", "17"],
        "scoring_command": [".venv/bin/python", "scripts/score_response_contract_generations.py", "--release", relative(release),
            "--spec", relative(output / "evaluation_spec.json"), "--generation-report",
            "runs/minicpm5_response_contract_v1_B1_dev16_20261022/generation_report.json", "--generations",
            "runs/minicpm5_response_contract_v1_B1_dev16_20261022/generations.jsonl", "--guard",
            "runs/minicpm5_response_contract_v1_B1_dev16_20261022_guard.json", "--output", relative(output / "native_B1_scoring.json")],
        "scoring_settings": {"grammar_module": contract["scoring"]["module"],
            "grammar_sha256": contract["scoring"]["sha256"], "unchanged": True,
            "single_line_compliance": contract["scoring"]["single_line_compliance"],
            "success_gate": contract["scoring"]["success_gate"], "task_correctness_separate": True},
        "scope": "Prompt-only follow-up to preserved failed native B1: same16 ordered dev worlds, same evidence/answers/model/template/greedy cap32 seed17 MPS FP32; family-only output instruction. Descriptive exposed development probe, no training/promotion or independent confirmation.",
        "unchanged_conditions": ["ordered case IDs", "source bytes and eligible evidence", "reference answers", "split roles",
            "source model and local tokenizer", "native chat template", "max_new_tokens32", "max_length512", "seed17",
            "greedy decoding and thinking disabled", "MPS FP32 with CPU fallback disabled", "strict whole-response parser"],
        "response_contract_delta": "Append two newlines plus a deterministic instruction derived only from public family/field-order grammar; target/action/value/support-count never choose the instruction."}
    write(output / "evaluation_spec.json", spec)
    return {"examples": 16, "output": relative(output), "data_sha256": spec["data_sha256"],
            "max_full_tokens": max(record["new"]["native_full"] for record in accounting),
            "preserved_failed_files": len(failed), "parent_spec_sha256": spec["parent_spec_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "parent-spec", "contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=ROOT / "checkpoints/minicpm5_source")
    parser.add_argument("--model-lock", type=Path, default=ROOT / "configs/minicpm5_sealed_lock.local.json")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    print(json.dumps(prepare(args.release, args.parent_spec, args.contract, args.output, tokenizer,
                             args.model_path, args.model_lock)))


if __name__ == "__main__":
    main()
