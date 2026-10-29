#!/usr/bin/env python3
"""Validate the density feasibility review packet without importing model code.

This is a preparation check, not a native launcher or source qualification. No
receipt accepted here can authorize execution. A dedicated runner and its local
execution closure must be reviewed and bound before native admission exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "configs/intrinsic_memory_density_protocol_v5.json"
PACKET = "reports/intrinsic_memory_density_v5/native_comparison_packet.json"
LAYOUTS = {"two": [3, 7], "full_attention_nine": list(range(3, 36, 4)), "all": list(range(36))}
GUARDS = {
    "driver_limit_bytes": 50 * 1024**3,
    "sustained_swap_growth_limit_bytes": 512 * 1024**2,
    "minimum_host_free_fraction": 0.15,
    "sustained_host_samples": 3,
    "exclusive_native_worker": True,
    "cpu_fallback": False,
}


def bound_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("binding path must be relative and contained")
    if "runtime_pilot.py" in path.parts:
        raise ValueError("protected source is not a density input")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("substituted binding path")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("binding escapes root")
    return current


def digest(root: Path, relative: str) -> str:
    path = bound_path(root, relative)
    if not path.is_file():
        raise ValueError(f"missing binding: {relative}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budgets() -> dict:
    # Eight complete-context worlds; five readout controls over eight worlds;
    # correction then revocation, contextual and exact, over eight new worlds.
    generations = 8 + 5 * 8 + 2 * 2 * 8
    tokens = 32
    prompt = 2048
    updates = 32 + 1
    # Parent reload canary and fresh-process pre-update parity add two forwards.
    nongenerative = updates + 2
    return {
        "competence_cases": 8,
        "training_updates": 32,
        "continuation_updates": 1,
        "primary_development_responses": 40,
        "chronology_responses": 32,
        "generation_requests_max": generations,
        "generated_tokens_max": generations * tokens,
        "forward_calls_max": generations * tokens + nongenerative,
        "backward_calls_max": updates,
        "optimizer_steps_max": updates,
        "processed_text_positions_max": generations * (tokens * prompt + tokens * (tokens - 1) // 2) + nongenerative * prompt,
        "backward_processed_positions_max": updates * prompt,
        "generation_sequence_tokens_max": prompt + tokens - 1,
        "model_loads_max": 2,
        "tokenizer_loads_max": 2,
        "P_updates": 0,
        "native_retries": 0,
        "wall_seconds_max": 7200,
        "artifact_bytes_max": 1024**3,
    }


def validate(protocol: dict, packet: dict, *, root: Path = ROOT) -> dict:
    if protocol.get("format") != "replica.intrinsic_memory.density.protocol.v5":
        raise ValueError("unsupported density protocol")
    if packet.get("format") != "replica.intrinsic_memory.density.review_packet.v5":
        raise ValueError("unsupported density packet")
    if any(packet.get(key) is not False for key in ("native_execution", "launch_allowed", "authorization")):
        raise ValueError("review packet cannot claim native admission")
    if packet.get("runner_status") != "not_implemented_native_admission_unavailable":
        raise ValueError("unreviewed runner status")
    if packet.get("protocol_sha256") != digest(root, PROTOCOL):
        raise ValueError("protocol binding differs")
    if protocol["placement"] != "block_input_query_to_post_block_residual":
        raise ValueError("undeclared placement experiment")
    shared = protocol["shared"]
    if (shared["selected_slots"] != 8 or shared["slot_width"] != 256
            or shared["slots_per_span"] != 4 or shared["active_P_experts"] != 1
            or shared["P_updates"] != 0 or shared["training_steps"] != 32
            or shared["decode"] != "greedy_eos_enabled_no_sampling"
            or shared["thinking_mode"] != "none" or shared["max_new_tokens"] != 32
            or shared["context_tokens_max"] != 2048):
        raise ValueError("unmatched shared exposure or memory contract")
    if shared["optimizer"] != "NativeFusedAdamW_one_owner_per_parameter":
        raise ValueError("optimizer ownership differs")
    if (shared["evidence_truncation_allowed"] is not False
            or shared["official_FP8"] is not False or shared["official_BF16"] is not False):
        raise ValueError("precision identity or complete evidence differs")
    scoring = protocol["scoring"]
    if scoring["competence"] != {"cases": 8, "correct_min": 8, "false_recall_max": 0,
            "evidence_truncation_max": 0, "incomplete": "stop_dependent_fit_evaluation"}:
        raise ValueError("competence gate differs")
    if scoring["primary"] != {"cases_per_control": 8, "contextual_correct_min": 6,
            "contextual_false_recall_max": 0, "exact_tolerance_cases": 1,
            "beats_each_negative_control": True}:
        raise ValueError("useful recall gate differs")
    if (scoring["chronology"]["worlds"] != 8
            or scoring["chronology"]["revoked_false_recall_max"] != 0
            or scoring["chronology"]["stale_snapshot_reads_accepted_max"] != 0
            or scoring["chronology"]["obsolete_retrieval_accepted_max"] != 0
            or scoring["chronology"]["obsolete_replay_accepted_max"] != 0):
        raise ValueError("chronology gate differs")
    if protocol["guards"] != GUARDS:
        raise ValueError("resource guard differs")
    bindings = protocol["bindings"]
    if len(bindings) < 12:
        raise ValueError("incomplete preparation bindings")
    for name, binding in bindings.items():
        if digest(root, binding["path"]) != binding["sha256"]:
            raise ValueError(f"preparation binding differs: {name}")
    if set(protocol["arms"]) != set(LAYOUTS) or set(packet["arms"]) != set(LAYOUTS):
        raise ValueError("incomplete density comparison")
    outputs = set()
    for name, sites in LAYOUTS.items():
        arm = protocol["arms"][name]
        if arm["sites"] != sites or arm["budgets"] != budgets():
            raise ValueError(f"layout or work budget differs: {name}")
        if arm["bindings"] != bindings or arm["guards"] != GUARDS:
            raise ValueError(f"per-arm execution/evidence/guard mismatch: {name}")
        if arm["shared_sha256"] != hashlib.sha256(json.dumps(shared, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
            raise ValueError("arm shared-contract binding differs")
        expected = hashlib.sha256(json.dumps(arm, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if packet["arms"][name]["arm_sha256"] != expected:
            raise ValueError("packet arm binding differs")
        if any(packet["arms"][name][key] != arm[key] for key in ("output_root", "guard_report")):
            raise ValueError("packet output or guard binding differs")
        if (packet["arms"][name]["authorized"] is not False
                or packet["arms"][name]["authorization_receipt"] != arm["authorization"]["receipt_path"]):
            raise ValueError("packet authorization binding differs")
        output = arm["output_root"]
        bound_path(root, output)
        if output in outputs:
            raise ValueError("arms share an output destination")
        outputs.add(output)
        if arm["authorization"]["required"] is not True or arm["authorization"]["inherited_from_v3"] is not False:
            raise ValueError("density authorization cannot inherit v3")
        if arm["numerical"] != protocol["numerical"] or arm["scoring"] != protocol["scoring"]:
            raise ValueError("per-arm numerical/scoring contract differs")
    total = {name: 3 * value for name, value in budgets().items()
             if name.endswith("_max") and name not in ("generation_sequence_tokens_max",)}
    if packet["comparison_budget_sum"] != total:
        raise ValueError("comparison budget sum differs")
    return {
        "status": "review_packet_consistent_native_not_admitted",
        "bound_inputs_checked": len(bindings),
        "arm_count": len(LAYOUTS),
        "per_arm_budgets": budgets(),
        "comparison_budget_sum": total,
        "native_execution": False,
        "source_weights_read": False,
        "model_imported": False,
        "authorization_accepted": False,
        "launch_allowed": False,
        "remaining_gates": packet["remaining_gates"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    protocol = json.loads(bound_path(args.root, PROTOCOL).read_text())
    packet = json.loads(bound_path(args.root, PACKET).read_text())
    print(json.dumps(validate(protocol, packet, root=args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
