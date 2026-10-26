#!/usr/bin/env python3
"""Reproduce the NL1 benchmark-validity findings without loading a model.

Reads only teaching/development/control evidence.  In particular, this command
never opens sealed_final.jsonl.  It is a diagnostic replay, not a new score or
benchmark admission.  The output is a fresh JSON artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from replica_lab.nested_learning import RouteControl
from replica_lab.nested_text import prompt_route_key

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/nested_text_nl1_20261026"
TEACH = ROOT / "runs/nested_text_nl1_20261026/teach02"
DEV04 = ROOT / "runs/nested_text_nl1_20261026/p_only_dev04/gates/dev.json"
ROLES = (
    "teaching", "development", "correction", "unaffected", "unknown",
    "revoked", "legacy_retention", "wrong_world", "wrong_world_teaching",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(role: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{role}.jsonl").read_text().splitlines()
            if line.strip()]


def encode_prompt(tokenizer, text: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "checkpoints/minicpm5_source", local_files_only=True,
        trust_remote_code=False)
    route = RouteControl("fixed", 16, 0)
    transactions = [json.loads(line) for line in
                    (TEACH / "transactions.jsonl").read_text().splitlines() if line.strip()]
    taught_rows = {row["id"]: row for row in rows("teaching") + rows("correction")}
    concept_experts: dict[str, set[int]] = {}
    for transaction in transactions:
        concept = taught_rows[transaction["event_id"]]["concept_or_rule_family"]
        concept_experts.setdefault(concept, set()).add(transaction["expert_id"])
    route_rows = []
    for row in rows("development"):
        ids = encode_prompt(tokenizer, row["prompt"])
        expert = route.route(prompt_route_key(torch.tensor([ids], dtype=torch.long)))
        taught = sorted(concept_experts.get(row["concept_or_rule_family"], []))
        route_rows.append({
            "id": row["id"], "concept": row["concept_or_rule_family"],
            "taught_experts": taught, "development_expert": expert,
            "matches_taught_concept": expert in taught,
        })

    lengths = []
    for role in ROLES:
        for row in rows(role):
            required = len(tokenizer(
                row["expected_response"], add_special_tokens=False)["input_ids"]) + 1
            lengths.append({"role": role, "id": row["id"], "required_with_eos": required})
    role_summary = {
        role: {
            "rows": sum(item["role"] == role for item in lengths),
            "over_8": sum(item["role"] == role and item["required_with_eos"] > 8
                          for item in lengths),
            "max_required_with_eos": max(
                (item["required_with_eos"] for item in lengths if item["role"] == role),
                default=0),
        }
        for role in ROLES
    }
    over_8 = [item for item in lengths if item["required_with_eos"] > 8]

    counterexamples = [
        {"expected": "43", "output": "The answer is 43."},
        {"expected": "43", "output": "43."},
        {"expected": "closes", "output": "The library closes at dusk."},
        {"expected": "Paris", "output": "Paris is the capital of France."},
    ]
    for case in counterexamples:
        case["current_exact_match"] = normalize(case["expected"]) == normalize(case["output"])
    completion = []
    for row in rows("legacy_retention"):
        if row["prompt"].lower().startswith("complete the sentence:"):
            completion.append({
                "id": row["id"], "prompt": row["prompt"],
                "expected": row["expected_response"],
                "has_missing_word": row["expected_response"].lower() not in row["prompt"].lower(),
            })

    teach_rows = sorted(rows("teaching") + rows("correction"),
                        key=lambda row: (float(row["target_available_at"]), row["id"]))[:27]
    positions = []
    for row in teach_rows:
        prompt_n = len(encode_prompt(tokenizer, row["prompt"]))
        response_n = len(tokenizer(
            row["expected_response"], add_special_tokens=False)["input_ids"]) + 1
        positions.append({
            "id": row["id"], "prompt_positions": prompt_n,
            "full_positions": prompt_n + response_n,
            "supervised_tokens": response_n,
            "two_pass_positions": 2 * prompt_n + response_n,
        })
    position_totals = {
        key: sum(row[key] for row in positions)
        for key in ("prompt_positions", "full_positions", "supervised_tokens", "two_pass_positions")
    }
    position_totals.update({
        "feature_forward_calls": 2 * len(positions),
        "site_feature_vectors": 2 * len(positions),
        "reported_processed_positions": json.loads((TEACH / "cost.json").read_text())["processed_positions"],
    })

    dev04 = json.loads(DEV04.read_text())
    groups = []
    for name, score in dev04["scores"].items():
        groups.append((f"taught:{name}", score))
    for name, score in dev04["preteach_scores"].items():
        groups.append((f"preteach:{name}", score))
    for name in ("shuffled_route", "reset_P", "spine_absent",
                 "wrong_world_P", "wrong_world_on_primary"):
        groups.append((f"control:{name}", dev04["controls"][name]))
    output_rows = []
    for group, score in groups:
        for row in score["results"]:
            text = row.get("output", "")
            output_rows.append({
                "group": group, "id": row["id"], "expected": row["expected"],
                "output": text,
                "tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
                "correct": row["correct"],
            })
    cap_summary: dict[str, dict[str, int]] = {}
    for row in output_rows:
        entry = cap_summary.setdefault(row["group"], {"rows": 0, "cap_8": 0})
        entry["rows"] += 1
        entry["cap_8"] += int(row["tokens"] == 8)
    taught_legacy = {row["id"]: row for row in dev04["scores"]["legacy_retention"]["results"]}
    preteach_legacy = {row["id"]: row for row in dev04["preteach_scores"]["legacy_retention"]["results"]}
    legacy = []
    for row_id in sorted(taught_legacy):
        taught, preteach = taught_legacy[row_id], preteach_legacy[row_id]
        legacy.append({
            "id": row_id, "expected": taught["expected"],
            "preteach_output": preteach["output"],
            "preteach_tokens": len(tokenizer(preteach["output"], add_special_tokens=False)["input_ids"]),
            "taught_output": taught["output"],
            "taught_tokens": len(tokenizer(taught["output"], add_special_tokens=False)["input_ids"]),
        })

    teach_cost = json.loads((TEACH / "cost.json").read_text())
    config = json.loads((ROOT / "configs/nested_text_nl1_v1.json").read_text())
    evidence_paths = [
        DATA / "manifest.json", DEV04, TEACH / "transactions.jsonl",
        TEACH / "cost.json", ROOT / "configs/nested_text_nl1_v1.json",
    ]
    result = {
        "format": "replica.nested_text.validity_reproduction.v1",
        "scope": "tokenizer-only read-only reproduction; not a corrected score or native admission",
        "native_model_executed": False,
        "tokenizer_loads": 1,
        "sealed_final_read": False,
        "evidence_sha256": {str(path.relative_to(ROOT)): sha(path) for path in evidence_paths},
        "routing": {
            "development_rows": len(route_rows),
            "matches_taught_concept": sum(row["matches_taught_concept"] for row in route_rows),
            "rows": route_rows,
        },
        "response_budget": {
            "current_max_new_tokens": 8,
            "expected_rows_over_8": len(over_8),
            "expected_role_summary": role_summary,
            "actual_scoring_attempts": len(output_rows),
            "generated_cap_8_rows": sum(row["tokens"] == 8 for row in output_rows),
            "cap_summary": cap_summary,
            "cap_examples": [row for row in output_rows if row["tokens"] == 8][:16],
            "note": "A cap hit alone does not prove truncation; visibly incomplete cap-hit outputs prove that some valid natural answers are truncated.",
        },
        "scorer_counterexamples": counterexamples,
        "sentence_completion_fixture": completion,
        "positions": position_totals,
        "position_rows": positions,
        "legacy_comparison": legacy,
        "duty_cycle_evidence": {
            "teaching_update_seconds": teach_cost["update_seconds"],
            "teaching_wall_seconds": teach_cost["wall_seconds"],
            "reported_duty_cycle": teach_cost["duty_cycle"],
            "gate_max_duty_cycle": config["gates"]["cost"]["max_duty_cycle"],
            "gate_has_amortized_workload_definition": "workload" in json.dumps(config["gates"]["cost"]),
        },
        "timing_code_evidence": {
            "request_preparation_before_timer": [
                "NestedTextLearner.plastic_read",
                "prompt_features/encode_sites",
                "MemorySession.prepare/sense encoding",
            ],
            "timer_covers": "generate_text only",
            "source": "scripts/run_nested_text.py:448-454",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "route_matches": result["routing"]["matches_taught_concept"],
        "generated_cap_8_rows": result["response_budget"]["generated_cap_8_rows"],
        "two_pass_positions": position_totals["two_pass_positions"],
        "reported_processed_positions": position_totals["reported_processed_positions"],
        "invalid_completion_rows": sum(not row["has_missing_word"] for row in completion),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
