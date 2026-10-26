#!/usr/bin/env python3
"""Recompute cached full-training phase costs and optimizer arithmetic offline."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "full_55": ["minicpm5_balanced_mask1_chunk01_report.json"],
    "full_256": [f"minicpm5_capacity256_chunk0{i}_report.json" for i in (1, 2)],
    "full_512": [f"minicpm5_capacity512_chunk0{i}_report.json" for i in (1, 2)],
    "full_augmented_256": [f"minicpm5_D2_capacity256_chunk0{i}_report.json" for i in (1, 2)],
    "full_augmented_512": [f"minicpm5_D2_capacity512_chunk0{i}_report.json" for i in (1, 2)],
}


def phase_updates(phases):
    updates = {}
    for event in phases:
        name = event["phase"]
        step = event.get("fields", {}).get("step")
        if step is None or name not in ("update_start", "backward_complete", "optimizer_step_complete"):
            continue
        row = updates.setdefault(step, {"step": step, "backward_calls": 0,
                                       "forward_backward_seconds": 0.0})
        if name == "update_start":
            row["start"] = event["elapsed_seconds"]
        elif name == "backward_complete":
            row["forward_backward_seconds"] += event["seconds_since_previous_mark"]
            row["backward_calls"] += 1
        else:
            row["post_backward_seconds"] = event["seconds_since_previous_mark"]
            row["total_seconds"] = event["elapsed_seconds"] - row["start"]
    return {step: row for step, row in updates.items() if "total_seconds" in row}


def analyze():
    source_hashes = {}
    cases = {}
    for label, names in CASES.items():
        updates = {}
        for name in names:
            path = ROOT / "reports" / name
            data = json.loads(path.read_text())
            source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            updates.update(phase_updates(data["profile"]["phases"]))
        rows = [updates[step] for step in sorted(updates)]
        total = sum(row["total_seconds"] for row in rows)
        tail = sum(row["post_backward_seconds"] for row in rows)
        fw = sum(row["forward_backward_seconds"] for row in rows)
        cases[label] = {
            "reports": ["reports/" + name for name in names], "updates": rows,
            "trainable_parameters": data["parameters"]["original"]["trainable"],
            "trainable_original_tensors": data["parameters"]["original"]["tensor_count"],
            "gradient_accumulation": data["grad_accum"],
            "actual_sequence_length": data["actual_sequence_lengths"],
            "mean_total_seconds": total / len(rows),
            "mean_forward_backward_seconds": fw / len(rows),
            "mean_post_backward_seconds": tail / len(rows),
            "post_backward_fraction": tail / total,
            "optimizer_updates_per_second": len(rows) / total,
            "conditional_projections_not_measurements": [
                {"post_backward_phase_speedup": factor,
                 "seconds_per_update": (total - tail + tail / factor) / len(rows),
                 "updates_per_second": len(rows) / (total - tail + tail / factor)}
                for factor in (2, 4, 8)
            ],
            "infinite_forward_backward_speedup_ceiling": total / tail,
        }
    reference = json.loads((ROOT / "reports/minicpm5_balanced_mask1_chunk01_report.json").read_text())
    shapes = {}
    for path in (ROOT / "checkpoints/minicpm5_source").glob("*.safetensors"):
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 16 * 1024 * 1024:
                raise ValueError("Unexpected unbounded safetensors header")
            header = json.loads(stream.read(length))
        shapes.update({name: value["shape"] for name, value in header.items() if name != "__metadata__"})
    shape_counts = collections.Counter()
    ns_flops = 0
    for group in reference["optimizer"]["groups"]:
        if group["algorithm"] != "muon8bit":
            continue
        for name in group["param_names"]:
            shape = tuple(shapes[name.removeprefix("base.")])
            short, long = sorted(shape)
            shape_counts[shape] += 1
            ns_flops += 5 * (4 * short * short * long + 2 * short ** 3)
    n = reference["parameters"]["original"]["total"]
    return {
        "format": "replica.m2_ultra.training_performance_analysis.v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hashes, "cached_cases": cases,
        "muon_arithmetic": {
            "scope": "Analytical GEMM count; not a measured kernel-time partition.",
            "shape_counts": [{"shape": list(shape), "tensors": count} for shape, count in sorted(shape_counts.items())],
            "newton_schulz_steps": 5, "gemms_per_matrix": 15,
            "gemms_per_update": sum(shape_counts.values()) * 15,
            "ns_flops_per_update": ns_flops,
            "formula_per_matrix": "5 * (4 * min(rows,cols)^2 * max(rows,cols) + 2 * min(rows,cols)^3)",
            "optimizer_finite_scalar_checks_per_update": 381 + sum(shape_counts.values()),
            "forward_backward_6NT_estimate_flops_55": 6 * n * 440,
            "note": "6NT is a coarse language-model FLOP approximation, excludes activation recomputation and detailed vocab/attention geometry; not measured useful hardware FLOPS.",
        },
        "memory_lower_bounds_bytes": {
            "weights_fp32": n * 4, "gradients_fp32": n * 4,
            "current_optimizer_state": reference["optimizer"]["state_tensor_bytes"],
            "current_weights_gradients_optimizer": n * 8 + reference["optimizer"]["state_tensor_bytes"],
            "all_adamw_fp32_weights_gradients_moments": n * 16,
            "excludes": "Activations, workspace, backend allocator, temporary tensors, live sources and host buffers.",
        },
        "limits": [
            "Post-backward markers include gradient audits, norm clipping and optimizer work; cached logs do not isolate their internal substeps.",
            "Warmup/allocation and process restart affect short runs; results are diagnostic.",
            "Performance gains need new scope-audited native tests; theoretical projections are not benchmark results.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing audit; choose a fresh path")
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(result["cached_cases"]),
                      "ns_TFLOP_per_update": result["muon_arithmetic"]["ns_flops_per_update"] / 1e12,
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
