#!/usr/bin/env python3
"""Independently score the v16 terminal mechanism criterion from raw outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from replica_lab.intrinsic_memory_budget_v12 import atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reconciliation", required=True, type=Path)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--reference-v12", required=True, type=Path)
    parser.add_argument("--reference-v14", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("content-key outcome analysis must be fresh")
    run = args.run.resolve()
    reconciliation = json.loads(args.reconciliation.read_text())
    probes = json.loads((run / "fit_probes_512.json").read_text())
    main_result = json.loads((run / "main_result.json").read_text())
    checkpoint = json.loads((run / "checkpoint/manifest.json").read_text())
    fit_rows = json.loads(args.fit.read_text())["rows"]
    answers = {row["world_id"]: row["answer"] for row in fit_rows}
    known_known = []
    for index in range(0, len(probes["rows"]), 2):
        pair = probes["rows"][index:index + 2]
        if all(answers[row["world_id"]] != "unknown" for row in pair):
            known_known.append(all(row["controls"]["contextual"]["correct"] for row in pair))
    if len(known_known) != 8:
        raise ValueError("known-known gate denominator differs from eight pairs")
    groups = [group["correct"] for group in probes["groups"]]
    groups_pass = all(group["fit_memory_dependence_passed"] for group in probes["groups"])
    known_pass = sum(known_known) > 1
    falsified = not groups_pass or not known_pass
    v12 = json.loads(args.reference_v12.read_text())
    v14 = json.loads(args.reference_v14.read_text())
    v12_correct = v12.get("correct", v12.get("fitting", {}).get("correct"))
    v12_pairs = v12.get("counterfactual_pairs_both_correct",
                        v12.get("fitting", {}).get("counterfactual_pairs_both_correct"))
    result = {
        "format": "replica.intrinsic_memory.content_key_outcome_analysis.v16",
        "packet_sha256": reconciliation["packet_sha256"],
        "protocol_sha256": reconciliation["protocol_sha256"],
        "run_root": str(run),
        "reader_key_mode": checkpoint.get("reader_key_mode"),
        "baseline_correct": reconciliation["baseline"]["correct"],
        "teacher_correct": reconciliation["teacher"]["correct"],
        "fitting_updates": reconciliation["fitting"]["updates"],
        "correct": reconciliation["fitting"]["correct"],
        "group_contextual_correct": groups,
        "all_final_groups_passed": groups_pass,
        "known_known_pairs_both_correct": sum(known_known),
        "known_known_pairs": len(known_known),
        "counterfactual_pairs_both_correct": reconciliation["fitting"][
            "counterfactual_pairs_both_correct"],
        "development_opened": reconciliation["development"]["opened"],
        "development_split_read_events": reconciliation["development"]["split_read_events"],
        "fresh_process_logits_exact": reconciliation["restart"]["fresh_process_logits_exact"],
        "actual_continuation_updates": reconciliation["restart"]["actual_continuation_updates"],
        "original_weights_unchanged": reconciliation["original_weights_unchanged"],
        "mechanism_falsified": falsified,
        "criterion": {
            "all_final_groups_required": True,
            "known_known_pairs_both_correct_must_exceed": 1,
            "passed": not falsified,
            "best_prefix_selection": False,
        },
        "comparison_with_v12_attempt10": {
            "contextual_delta": reconciliation["fitting"]["correct"]["contextual"] - v12_correct["contextual"],
            "shuffled_delta": reconciliation["fitting"]["correct"]["shuffled"] - v12_correct["shuffled"],
            "pairs_delta": reconciliation["fitting"]["counterfactual_pairs_both_correct"] - v12_pairs,
            "reference_receipt": str(args.reference_v12),
        },
        "comparison_with_pre_mlp_v14": {
            "contextual_delta": reconciliation["fitting"]["correct"]["contextual"] - v14["correct"]["contextual"],
            "shuffled_delta": reconciliation["fitting"]["correct"]["shuffled"] - v14["correct"]["shuffled"],
            "pairs_delta": reconciliation["fitting"]["counterfactual_pairs_both_correct"] - v14["counterfactual_pairs_both_correct"],
            "reference_receipt": str(args.reference_v14),
        },
        "resources": {
            "supervision_elapsed_seconds": reconciliation["termination"]["elapsed_seconds"],
            **reconciliation["resources"],
        },
        "capability_boundary": {
            "useful_recall": reconciliation["outcome"]["useful_recall"],
            "development_generalization_measured": reconciliation["outcome"][
                "development_generalization_measured"],
            "fresh_process_continuation_passed": reconciliation["outcome"][
                "fresh_process_continuation_passed"],
            "formation_persistent_learning_density_promotion_RSI": False,
            "less_compressed_or_compression_training_combined": False,
            "exact_evidence_is_charged_reference": True,
        },
        "raw_reconciliation": str(args.reconciliation),
        "references_not_rerun": [str(args.reference_v12), str(args.reference_v14)],
    }
    if main_result.get("reader_key_mode") != "content":
        raise ValueError("native result reader key mode differs")
    if result["reader_key_mode"] != "content":
        raise ValueError("content-key checkpoint identity missing")
    if result["fitting_updates"] != 512:
        raise ValueError("terminal analysis requires exactly 512 fitting updates")
    if falsified and result["development_opened"]:
        raise ValueError("development opened after mechanism falsification")
    atomic(args.output.resolve(), result)
    print(json.dumps({
        "correct": result["correct"], "groups": groups,
        "known_known_pairs_both_correct": result["known_known_pairs_both_correct"],
        "mechanism_falsified": falsified,
        "development_opened": result["development_opened"],
        "fresh_process_continuation_passed": result["capability_boundary"][
            "fresh_process_continuation_passed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
