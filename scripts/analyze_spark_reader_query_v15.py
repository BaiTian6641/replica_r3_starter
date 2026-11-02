#!/usr/bin/env python3
"""Independently analyze preserved v15 reader/query artifacts; no model execution."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import torch
from safetensors.torch import load_file

from replica_lab.intrinsic_memory_budget_v12 import atomic
from replica_lab.intrinsic_memory_reader_query_v15 import CONTROLS

LABELS = ("red", "blue", "green", "gold")
INDEX = {label: index for index, label in enumerate(LABELS)}
REFERENCE_CONTROLS = (0, 3, 2, 4)


def probe(features, answers, *, shuffled=None, identities=None):
    from replica_lab.intrinsic_memory_probe_v18 import probe as grouped_probe
    return grouped_probe(features, answers, shuffled=shuffled, identities=identities,
                         solver="dual")


def probe_summary(features, answers, identities):
    baseline = probe(features, answers, identities=identities)
    controls = []
    for seed in range(100):
        permutation = list(range(len(answers)))
        random.Random(20261030 + seed).shuffle(permutation)
        controls.append(sum(row["correct"] for row in probe(
            features, answers, shuffled=permutation, identities=identities)))
    return {
        "baseline_correct": sum(row["correct"] for row in baseline), "total": len(baseline),
        "baseline_predictions": baseline,
        "shuffled_label_controls": {
            "count": len(controls), "correct_min": min(controls), "correct_max": max(controls),
            "correct_mean": sum(controls) / len(controls), "scores": controls,
        },
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(run, reference):
    run, reference = Path(run), Path(reference)
    answers, actual_first, identities = [], [], []
    known_known_first, known_known_pairs_correct = [], 0
    features = {name: [] for name in (
        "slots_mean", "weights_3", "weights_7", "read_3", "read_7", "query_3", "query_7")}
    concentration = {site: {control: [] for control in CONTROLS} for site in (3, 7)}
    top_slots = {site: Counter() for site in (3, 7)}
    pair_top_same = {site: 0 for site in (3, 7)}
    pair_weight_l1 = {site: [] for site in (3, 7)}
    pair_read_mean_abs = {site: [] for site in (3, 7)}
    exact_logits, wrong_partner = True, {site: 0.0 for site in (3, 7)}
    max_logit_delta = 0.0
    artifact_hashes = {}
    for index in range(16):
        document_path = run / "pairs" / f"pair_{index:02d}.json"
        document = json.loads(document_path.read_text())
        tensor_path = run / "pairs" / f"pair_{index:02d}.safetensors"
        if sha256(tensor_path) != document["tensor_artifact_sha256"]:
            raise ValueError("reader/query tensor hash differs")
        artifact_hashes[str(tensor_path.relative_to(run))] = document["tensor_artifact_sha256"]
        tensors = load_file(str(tensor_path))
        old = load_file(str(reference / "pairs" / f"pair_{index:02d}.safetensors"))
        for new_index, old_index in enumerate(REFERENCE_CONTROLS):
            delta = float((tensors["logits"][:, new_index].float()
                           - old["logits"][:, old_index].float()).abs().max())
            max_logit_delta = max(max_logit_delta, delta)
            exact_logits = exact_logits and delta == 0
        for site in (3, 7):
            weights = tensors[f"site_{site}_weights"].float()
            read = tensors[f"site_{site}_read"].float()
            pair_weight_l1[site].append(float((weights[0, 0] - weights[1, 0]).abs().sum()))
            pair_read_mean_abs[site].append(float((read[0, 0] - read[1, 0]).abs().mean()))
            pair_top_same[site] += int(int(weights[0, 0].argmax()) == int(weights[1, 0].argmax()))
            wrong_partner[site] = max(
                wrong_partner[site], float((weights[0, 1] - weights[1, 0]).abs().max()),
                float((weights[1, 1] - weights[0, 0]).abs().max()))
            for side in (0, 1):
                for control_index, control in enumerate(CONTROLS):
                    current = weights[side, control_index]
                    concentration[site][control].append({
                        "entropy": float(-(current * current.clamp_min(1e-12).log()).sum()),
                        "max_weight": float(current.max()),
                    })
                top = int(weights[side, 0].argmax())
                top_slots[site][top] += 1
        if all(answer != "unknown" for answer in document["expected_answers"]):
            flags = [document["control_outputs"][side][0]["first_token_correct"] for side in (0, 1)]
            known_known_first.extend(flags)
            known_known_pairs_correct += int(all(flags))
        for side in (0, 1):
            answer = document["expected_answers"][side]
            if answer == "unknown":
                continue
            answers.append(answer)
            identities.append({"pair_index": index, "side": side,
                               "world_id": document["world_ids"][side]})
            actual_first.append(document["control_outputs"][side][0]["first_token_correct"])
            slots = tensors[f"slots_{side}"].float()
            averaged = slots.mean(dim=0)
            features["slots_mean"].append(averaged / averaged.norm().clamp_min(1e-12))
            for site in (3, 7):
                features[f"weights_{site}"].append(tensors[f"site_{site}_weights"][side, 0].float())
                features[f"read_{site}"].append(tensors[f"site_{site}_read"][side, 0].float())
                features[f"query_{site}"].append(tensors[f"site_{site}_query"][side, 0].float())
    concentration_summary = {}
    for site in (3, 7):
        concentration_summary[str(site)] = {}
        for control in CONTROLS:
            rows = concentration[site][control]
            concentration_summary[str(site)][control] = {
                "entropy_mean": statistics.mean(row["entropy"] for row in rows),
                "entropy_min": min(row["entropy"] for row in rows),
                "entropy_max": max(row["entropy"] for row in rows),
                "max_weight_mean": statistics.mean(row["max_weight"] for row in rows),
            }
    return {
        "format": "replica.intrinsic_memory.reader_query_analysis.v15.qualification2",
        "run_root": str(run), "reference_root": str(reference),
        "rows": 32, "pairs": 16, "known_answer_rows": len(answers),
        "exact_reference_first_token_logits": exact_logits,
        "max_reference_logit_absolute_delta": max_logit_delta,
        "wrong_partner_weight_max_absolute_delta": wrong_partner,
        "attention_concentration": concentration_summary,
        "top_slot_counts": {str(site): dict(top_slots[site]) for site in (3, 7)},
        "counterfactual_pairs_with_same_top_slot": pair_top_same,
        "counterfactual_weight_l1": {
            str(site): {"mean": statistics.mean(pair_weight_l1[site]),
                        "min": min(pair_weight_l1[site]), "max": max(pair_weight_l1[site])}
            for site in (3, 7)},
        "counterfactual_read_mean_absolute": {
            str(site): {"mean": statistics.mean(pair_read_mean_abs[site]),
                        "min": min(pair_read_mean_abs[site]), "max": max(pair_read_mean_abs[site])}
            for site in (3, 7)},
        "known_answer_first_token_correct": sum(actual_first),
        "known_known_first_token_correct": sum(known_known_first),
        "known_known_rows": len(known_known_first),
        "known_known_pairs_both_correct": known_known_pairs_correct,
        "decodability": {name: probe_summary(torch.stack(rows), answers, identities)
                         for name, rows in features.items()},
        "selection": {
            "primary_mechanism": "historical_v16_selection_preserved_no_new_selection",
            "hypothesis": "learned-order-dominated keys cause positional addressing collapse despite information-preserving reads",
            "falsification": "reject if any final eight-world fitting group fails or known-known pairs both correct do not exceed 1/8 after exactly 512 updates",
            "source_features_changed": False, "slot_count_changed": False,
            "ordered_values_changed": False, "placement_changed": False,
            "objective_changed": False, "optimizer_changed": False,
            "less_compressed_or_compression_training_combined": False,
            "exact_evidence_is_charged_reference_not_learning": True,
        },
        "artifact_sha256": artifact_hashes,
        "scope": {
            "fitting_only": True, "development_labels_used": False,
            "parameter_updates": 0, "native_model_executed_by_analysis": False,
            "held_out_generalization": False, "calibrated_confidence": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("reader/query analysis receipt must be fresh")
    result = analyze(args.run.resolve(), args.reference.resolve())
    atomic(args.output.resolve(), result)
    print(json.dumps({
        "exact_reference_logits": result["exact_reference_first_token_logits"],
        "top_slot_counts": result["top_slot_counts"],
        "pairs_same_top_slot": result["counterfactual_pairs_with_same_top_slot"],
        "decodability": {name: value["baseline_correct"]
                         for name, value in result["decodability"].items()},
        "selection": result["selection"]["primary_mechanism"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
