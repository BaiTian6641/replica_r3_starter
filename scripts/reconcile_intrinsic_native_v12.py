#!/usr/bin/env python3
"""Independently reconcile a terminal v12 native run from raw receipts only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from replica_lab.intrinsic_memory_budget_v12 import WorkLedger, atomic


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def response_summary(document):
    rows = document["rows"]
    return {
        "rows": len(rows), "complete": document.get("complete"),
        "correct": sum(row["correct"] for row in rows),
        "false_recall": sum(row["false_recall"] for row in rows),
        "terminated": sum(row["terminated"] for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
        "all_content_and_responses_complete": all(
            row["correct"] and not row["false_recall"]
            and row["terminated"] and not row["truncated"] for row in rows),
    }


def reconcile(run: Path, packet: Path, protocol_path: Path):
    main = load(run / "main_result.json")
    baseline = response_summary(load(run / "baseline.json"))
    teacher_document = load(run / "teacher_qualification.json")
    teacher = response_summary(teacher_document)
    probes = load(run / "fit_probes_512.json")
    controls = sorted(probes["correct"])
    recomputed = {control: sum(row["controls"][control]["correct"]
                               for row in probes["rows"]) for control in controls}
    if recomputed != probes["correct"]:
        raise ValueError("final control totals differ from raw rows")
    groups = [{"index": index, "correct": group["correct"],
               "counterfactual_pairs_both_correct": group["counterfactual_pairs_both_correct"],
               "fit_memory_dependence_passed": group["fit_memory_dependence_passed"]}
              for index, group in enumerate(probes["groups"])]
    pair_rows = []
    for index in range(0, len(probes["rows"]), 2):
        sides = []
        for row in probes["rows"][index:index + 2]:
            contextual = row["controls"]["contextual"]
            sides.append({
                "world_id": row["world_id"], "round_trip_exact": row["round_trip_exact"],
                "raw_answer": contextual["raw_answer"],
                "content_correct": contextual["correct"],
                "first_token_correct": contextual["first_token_correct"],
                "terminated": contextual["terminated"], "truncated": contextual["truncated"],
                "prefix_logits_exact": contextual["fit_generation_prefix_logits_exact"],
                "logit_max_delta_from_correct": contextual["logit_max_delta_from_correct"],
                "control_correct": {control: row["controls"][control]["correct"]
                                    for control in controls},
                "reader_layers": row["layers"],
            })
        pair_rows.append({"pair": [side["world_id"] for side in sides],
                          "both_contextual_correct": all(
                              side["content_correct"] for side in sides), "sides": sides})
    if sum(pair["both_contextual_correct"] for pair in pair_rows) != probes["counterfactual_pairs_both_correct"]:
        raise ValueError("counterfactual pair total differs from raw rows")
    control_completion = {control: {
        "terminated": sum(row["controls"][control]["terminated"] for row in probes["rows"]),
        "truncated": sum(row["controls"][control]["truncated"] for row in probes["rows"]),
        "first_token_correct": sum(row["controls"][control]["first_token_correct"]
                                   for row in probes["rows"]),
        "prefix_logits_exact": sum(row["controls"][control]["fit_generation_prefix_logits_exact"]
                                   for row in probes["rows"]),
    } for control in controls}
    with (run / "work_ledger.jsonl").open("rb") as stream:
        identity = json.loads(stream.readline())["identity"]
    ledger = WorkLedger(run / "work_ledger.jsonl", identity=identity).snapshot()
    if any(event["status"] == "reserved" for event in ledger["events"]):
        raise ValueError("terminal ledger contains uncertain work")
    development_reads = [event for event in ledger["events"]
                         if event["operation"] == "development_split_read"]
    guard_main = load(run / "capacity_guard_main.json")
    guard_restart = load(run / "capacity_guard_restart.json")
    supervision = load(run / "supervision/receipt.json")
    failure = load(run / "restart.failure.json")
    checkpoint = load(run / "checkpoint/manifest.json")
    artifact_bytes = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    final_fit_passed = (main["fit_updates"] == 512
                        and all(group["fit_memory_dependence_passed"] for group in probes["groups"]))
    if final_fit_passed != main["fit_memory_dependence_passed"]:
        raise ValueError("main fitting result differs from final groups")
    development_opened = bool(development_reads) or (run / "development.json").exists()
    if development_opened and not final_fit_passed:
        raise ValueError("development opened after failed final fitting")
    raw_names = ["baseline.json", "teacher_qualification.json", "fit_probes_512.json",
                 "main_result.json", "restart.failure.json", "work_ledger.jsonl",
                 "capacity_guard_main.json", "capacity_guard_restart.json",
                 "supervision/receipt.json", "checkpoint/manifest.json"]
    return {
        "format": "replica.intrinsic_memory.terminal_reconciliation.v12",
        "run_root": str(run), "packet_path": str(packet),
        "packet_sha256": sha256(packet),
        "protocol_path": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "preflight_packet_sha256": load(run / "preflight.json")["packet_sha256"],
        "raw_sha256": {name: sha256(run / name) for name in raw_names},
        "baseline": baseline,
        "teacher": {**teacher, "passed_flag": teacher_document["passed"],
                    "frozen_original_extensions_disabled": teacher_document[
                        "frozen_original_extensions_disabled"]},
        "fitting": {
            "updates": main["fit_updates"], "fit_memory_dependence_passed": final_fit_passed,
            "correct": recomputed, "groups": groups,
            "counterfactual_pairs_both_correct": probes["counterfactual_pairs_both_correct"],
            "counterfactual_pairs": pair_rows,
            "round_trip_exact": sum(row["round_trip_exact"] for row in probes["rows"]),
            "control_completion": control_completion,
            "selection": "fixed_final_512_all_groups_only_no_best_prefix",
        },
        "development": {"opened": development_opened,
                        "split_read_events": len(development_reads),
                        "required_to_remain_closed": not final_fit_passed},
        "termination": {
            "supervisor_status": supervision["status"], "exit_code": supervision["exit_code"],
            "elapsed_seconds": supervision["elapsed_seconds"],
            "deadline_overrun_seconds": supervision["deadline_overrun_seconds"],
            "process_group_exit_confirmed": supervision["process_group_exit_confirmed"],
        },
        "restart": {
            "requested": main["restart_requested"], "failure_type": failure["type"],
            "failure_message": failure["message"],
            "extension_restore_completed": any(
                event["operation"] == "extension_optimizer_restore"
                and event["status"] == "completed" for event in ledger["events"]),
            "fresh_process_logits_exact": False, "actual_continuation_updates": 0,
            "fresh_process_continuation_measured": False,
        },
        "checkpoint": {"contains_base_weights": checkpoint["contains_base_weights"],
                       "original_weights_serialized": checkpoint["original_weights_serialized"],
                       "producer_release": checkpoint["producer_release"],
                       "manifest_sha256": sha256(run / "checkpoint/manifest.json")},
        "original_weights_unchanged": main["original_weights_unchanged"],
        "resources": {
            "peak_driver_bytes": max(guard_main["peak_driver_bytes"],
                                     guard_restart["peak_driver_bytes"]),
            "peak_swap_growth_bytes": max(guard_main["peak_swap_growth_bytes"],
                                          guard_restart["peak_swap_growth_bytes"]),
            "minimum_observed_free_percent": min(guard_main["minimum_observed_free_percent"],
                                                 guard_restart["minimum_observed_free_percent"]),
            "artifact_bytes_observed_by_guard": guard_restart["artifact_bytes_observed"],
            "artifact_bytes_current_raw_tree": artifact_bytes,
            "driver_and_rss_overlap": True, "samples_are_not_hard_realtime_maxima": True,
        },
        "ledger": {"reserved": ledger["reserved"], "journal": ledger["journal"],
                   "events": len(ledger["events"]),
                   "active_stages": [name for name, state in ledger["stages"].items()
                                     if state["status"] == "active"]},
        "diagnostic_limits": {
            "raw_source_features_preserved": False, "compressed_slots_preserved": False,
            "full_first_token_logit_vectors_preserved": False,
            "available_per_pair_evidence": "answers, correctness, reader RMS/gates and logit max deltas",
            "next_required_measurement": "native fitting-only source-feature/slot/reader/residual/full-logit diagnostic",
        },
        "outcome": {
            "native_competence": main["competence_passed"],
            "final_fitting_passed": final_fit_passed,
            "useful_recall": False, "development_generalization_measured": False,
            "fresh_process_continuation_passed": False,
            "persistent_learning": False, "promotion": False, "RSI": False,
            "status": "final_fit_failed_development_closed_restart_execution_failed",
        },
        "analysis_written_outside_raw_run": True,
    }


def main_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("reconciliation receipt must be fresh")
    result = reconcile(args.run.resolve(), args.packet.resolve(), args.protocol.resolve())
    if result["packet_sha256"] != result["preflight_packet_sha256"]:
        raise ValueError("packet differs from the admitted attempt")
    atomic(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
