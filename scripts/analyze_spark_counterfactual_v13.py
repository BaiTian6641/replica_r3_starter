#!/usr/bin/env python3
"""Independently analyze v13 pair evidence and select one successor mechanism."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from replica_lab.intrinsic_memory_budget_v12 import atomic

CONTROLS = ("contextual", "row_permutation", "shuffled", "wrong", "zero", "absent", "exact_text")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt())


def analyze(run: Path):
    summary = json.loads((run / "diagnostic.json").read_text())
    pairs = []
    rms_values = []
    for index in range(16):
        path = run / "pairs" / f"pair_{index:02d}.json"
        document = json.loads(path.read_text())
        tensors = load_file(str(run / "pairs" / f"pair_{index:02d}.safetensors"))
        contextual = document["controls"].index("contextual")
        sides = []
        for side in (0, 1):
            output = document["control_outputs"][side][contextual]
            reader_rms = rms(tensors["reader"][side, contextual])
            projection_rms = rms(tensors["projection"][side, contextual])
            residual_rms = rms(tensors["residual"][side, contextual])
            rms_values.append((reader_rms, projection_rms, residual_rms))
            sides.append({
                "world_id": document["world_ids"][side], "raw_answer": output["raw_answer"],
                "content_correct": output["correct"],
                "expected_minus_other_logit": output["expected_minus_other_logit"],
                "first_token_correct": output["first_token_correct"],
                "reader_rms": reader_rms, "projection_rms": projection_rms,
                "residual_rms": residual_rms,
            })
        def max_change(control):
            return max(value["max_absolute_logits"]
                       for value in document["control_logit_changes"][control])
        pairs.append({
            "pair_index": index, "world_ids": document["world_ids"],
            "both_contextual_correct": document["both_contextual_correct"],
            "source_feature_distance": document["source_feature_distance"],
            "compressed_slot_distance": document["compressed_slot_distance"],
            "reader_distance": document["reader_distance"],
            "residual_distance": document["residual_distance"],
            "contextual_distribution_distance": document["contextual_distribution_distance"],
            "row_permutation_max_logit_change": max_change("row_permutation"),
            "slot_shuffle_max_logit_change": max_change("shuffled"),
            "wrong_source_max_logit_change": max_change("wrong"),
            "sides": sides,
        })
    # Recompute directly from source documents rather than the summary receipt.
    raw_documents = [json.loads((run / "pairs" / f"pair_{index:02d}.json").read_text())
                     for index in range(16)]
    recomputed = {control: sum(document["control_outputs"][side][control_index]["correct"]
                               for document in raw_documents for side in (0, 1))
                  for control_index, control in enumerate(CONTROLS)}
    if recomputed != summary["correct"]:
        raise ValueError("diagnostic control totals differ from pair receipts")
    for control in CONTROLS:
        for field in ("terminated", "truncated"):
            observed = sum(document["control_outputs"][side][index][field]
                           for document in raw_documents for side in (0, 1)
                           for index, name in enumerate(CONTROLS) if name == control)
            if observed != summary["complete_responses"][control][field]:
                raise ValueError("diagnostic completion totals differ")
    rms_tensor = torch.tensor(rms_values)
    result = json.loads((run / "result.json").read_text())
    return {
        "format": "replica.intrinsic_memory.counterfactual_decision.v13",
        "run_root": str(run), "diagnostic_sha256": sha256(run / "diagnostic.json"),
        "result_sha256": sha256(run / "result.json"),
        "pair_sha256": {f"pair_{index:02d}": {
            "json": sha256(run / "pairs" / f"pair_{index:02d}.json"),
            "safetensors": sha256(run / "pairs" / f"pair_{index:02d}.safetensors"),
        } for index in range(16)},
        "controls": list(CONTROLS), "correct": recomputed,
        "counterfactual_pairs_both_correct": sum(pair["both_contextual_correct"] for pair in pairs),
        "complete_responses": summary["complete_responses"],
        "stage_checks": {
            "all_pairs_have_direct_slot_comparison": all(
                pair["compressed_slot_distance"]["direct_comparison"] for pair in pairs),
            "minimum_slot_max_absolute_distance": min(
                pair["compressed_slot_distance"]["max_absolute"] for pair in pairs),
            "minimum_reader_max_absolute_distance": min(
                min(site["max_absolute"] for site in pair["reader_distance"]) for pair in pairs),
            "minimum_residual_max_absolute_distance": min(
                min(site["max_absolute"] for site in pair["residual_distance"]) for pair in pairs),
            "source_feature_shape_mismatch_pairs": [
                pair["pair_index"] for pair in pairs
                if not pair["source_feature_distance"]["direct_comparison"]],
            "mean_reader_rms": float(rms_tensor[:, 0].mean()),
            "mean_projection_rms": float(rms_tensor[:, 1].mean()),
            "mean_residual_rms": float(rms_tensor[:, 2].mean()),
        },
        "control_interpretation": {
            "complete_row_permutation": "Moves slot content and its learned address together; near-invariance is expected up to numerical reduction order.",
            "sealed_slot_shuffle": "Reassigns slot content to original addresses; historical v12 gate interpretation is unchanged.",
            "semantic_wrong_source": "Changes the answer-bearing source fact rather than only addresses.",
            "exact_text": "Charged exact-evidence reference, not parametric learning evidence.",
        },
        "pairs": pairs,
        "resources": result["capacity"],
        "artifact_bytes_observed": result["artifact_bytes_observed"],
        "reservations": result["reservations"],
        "original_weights_unchanged": result["original_weights_unchanged"],
        "development_labels_used": False, "parameter_updates": 0,
        "decision": {
            "mechanism": "pre_mlp_fusion_placement_single_change",
            "source_representation": "keep_the_same_frozen_features_and_four_slot_compressor",
            "reader_sites": "keep_3_and_7_for_the_first_comparison",
            "reason": "Every pair retains nonzero slot and reader distinctions, while residual distinctions are smaller and eight final counterfactual pairs still have an incorrect side. Compression-loss is therefore not the decisive first hypothesis. Test whether fusion placement changes answer use before changing capacity.",
            "do_not_change": ["slot_count", "source_encoder", "reader_sites", "objective", "development_labels", "optimizer"],
            "less_compressed_arm_selected": False,
            "reader_query_alignment_fallback_if_placement_fails": True,
            "causal_benefit_established_by_this_diagnostic": False,
        },
        "limitations": [
            "Thirty-two exposed fitting rows only; development remains unopened.",
            "Nonzero distances show retained distinctions, not sufficient information or calibrated confidence.",
            "Reader/fusion causality requires the preregistered placement comparison.",
            "Two fitting rule families and one seed do not establish broad generalization.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("decision receipt must be fresh")
    result = analyze(args.run.resolve())
    atomic(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
