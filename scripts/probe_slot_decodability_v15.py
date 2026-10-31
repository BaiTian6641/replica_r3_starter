#!/usr/bin/env python3
"""Bounded within-fitting-set slot decodability probe; no generalization claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from safetensors.torch import load_file

from replica_lab.intrinsic_memory_budget_v12 import atomic

LABELS = ("red", "blue", "green", "gold")
INDEX = {label: index for index, label in enumerate(LABELS)}


def probe(features, answers, *, shuffled=None):
    one_hot = torch.zeros(len(answers), len(LABELS))
    for index, answer in enumerate(answers):
        one_hot[index, INDEX[answer]] = 1.0
    if shuffled is not None:
        one_hot = one_hot[shuffled]
    predictions = []
    for holdout in range(len(features) // 2):
        train = torch.tensor([index for index in range(len(features))
                              if index // 2 != holdout])
        center = features[train].mean(dim=0)
        prior = one_hot[train].mean(dim=0)
        x = features[train] - center
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        y = one_hot[train] - prior
        weights = torch.linalg.solve(x.T @ x + torch.eye(x.shape[1]), x.T @ y)
        for index in (holdout * 2, holdout * 2 + 1):
            normalized = features[index] - center
            normalized = normalized / normalized.norm().clamp_min(1e-12)
            scores = normalized @ weights + prior
            predicted = LABELS[int(scores.argmax())]
            predictions.append({"world_index": index, "actual": answers[index],
                                "predicted": predicted, "correct": predicted == answers[index]})
    return predictions
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--controls", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("decodability receipt must be fresh")
    features, answers = [], []
    artifact_hashes = {}
    for index in range(16):
        path = args.run / "pairs" / f"pair_{index:02d}.safetensors"
        tensors = load_file(str(path))
        artifact_hashes[path.name] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        for side in (0, 1):
            document = json.loads((args.run / "pairs" / f"pair_{index:02d}.json").read_text())
            answer = document["expected_answers"][side]
            if answer == "unknown":
                continue
            slots = tensors[f"slots_{side}"].float()
            averaged = slots.mean(dim=0)
            averaged = averaged / averaged.norm().clamp_min(1e-12)
            features.append(averaged)
            answers.append(answer)
    features = torch.stack(features)
    baseline = probe(features, answers)
    controls = []
    for seed in range(args.controls):
        permutation = list(range(len(answers)))
        random.Random(20261030 + seed).shuffle(permutation)
        controls.append(probe(features, answers, shuffled=permutation))
    result = {
        "format": "replica.intrinsic_memory.slot_decodability_probe.v15",
        "run_root": str(args.run),
        "rows": len(answers), "pairs": len(answers) // 2,
        "feature": "mean of four V13 slots, per-row L2 normalization, training-fold centering",
        "model": "ridge regression alpha=1 with unpenalized training-label-prior intercept",
        "evaluation": "whole-pair holdouts on known-answer colors only",
        "baseline": {"correct": sum(row["correct"] for row in baseline), "total": len(baseline),
                     "predictions": baseline},
        "shuffled_label_controls": {
            "count": len(controls),
            "correct_min": min(sum(row["correct"] for row in control) for control in controls),
            "correct_max": max(sum(row["correct"] for row in control) for control in controls),
            "correct_mean": sum(sum(row["correct"] for row in control) for control in controls) / len(controls),
            "scores": [sum(row["correct"] for row in control) for control in controls],
        },
        "artifact_sha256": artifact_hashes,
        "scope": {
            "within_fitting_set_decodability_only": True,
            "compressor_trained_on_these_rows": True,
            "held_out_generalization": False,
            "complete_fact_preservation": False,
            "native_reader_uses_information": False,
            "development_labels_used": False,
            "parameter_updates": 0,
        },
    }
    atomic(args.output, result)
    print(json.dumps({"baseline": result["baseline"]["correct"],
                      "controls": result["shuffled_label_controls"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
