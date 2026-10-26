#!/usr/bin/env python3
"""Reproduce a scope-preserving Spark compute budget from pinned evidence.

Standard library only. Reads small research snapshots; never imports a model,
loads tensor payloads, starts a GPU process or provisions paid resources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/sparkx_formal_training_budget_estimate_v1.json"


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(folder):
    manifest = read_json(folder / "input_manifest.json")
    for item in manifest["inputs"]:
        path = folder / item["snapshot"]
        if sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Evidence snapshot differs: {item['snapshot']}")
    return manifest


def flop_budget(config, length, window_aware=True):
    workload = config["workload"]
    base = workload["checkpointed_training_parameter_flop_factor"] * config["model"]["backbone_parameters"]
    if window_aware:
        spans = workload["full_attention_layers"] * length + workload["sliding_attention_layers"] * min(length, workload["sliding_window"])
    else:
        spans = workload["layers"] * length
    attention = workload["checkpointed_attention_flop_factor"] * workload["query_heads"] * workload["head_dim"] * spans
    reserves = {key: workload[key] for key in (
        "live_encoder_flops_reserve_per_position", "native_extension_flops_reserve_per_position",
        "unimplemented_branch_and_non_gemm_flops_reserve_per_position")}
    return {"sequence_length": length, "window_aware_attention": window_aware,
            "backbone_flops_per_position": base, "attention_flops_per_position": attention,
            "assumed_reserves": reserves, "total_flops_per_position": base + attention + sum(reserves.values())}


def hours_for_positions(positions, rates, gpus, overhead):
    if positions <= 0 or not 0 < rates[0] <= rates[1] or gpus < 1:
        raise ValueError("Positive positions, ordered positive rates and GPU count required")
    wall = [positions / rates[1] / 3600 * overhead[0], positions / rates[0] / 3600 * overhead[1]]
    return {"wall_hours": wall, "gpu_hours": [hours * gpus for hours in wall],
            "wall_days_at_continuous_use": [hours / 24 for hours in wall]}


def m2_calibration(config, inputs):
    sustained = read_json(inputs / "fused_sustained_report.json")
    cached = read_json(inputs / "minicpm_cached_analysis.json")
    params = sustained["trainable_parameters"]
    steady = sustained["performance"]["steady"]
    checks = steady["mean_seconds_by_phase"]
    samples = []
    for key in ("full_55", "full_256", "full_512"):
        case = cached["cached_cases"][key]
        lengths = case["actual_sequence_length"]
        if lengths["minimum"] != lengths["maximum"]:
            raise ValueError("Calibration requires the observed uniform lengths")
        positions = sum(row["backward_calls"] for row in case["updates"]) * lengths["minimum"] / len(case["updates"])
        seconds = case["mean_forward_backward_seconds"]
        samples.append({"case": key, "positions_per_update": positions, "forward_backward_seconds": seconds,
                        "checkpointed_equivalent_TFLOPS": 8 * params * positions / seconds / 1e12})
    samples.append({"case": "fused_steady_55", "positions_per_update": 440,
                    "forward_backward_seconds": checks["forward_backward_seconds"],
                    "checkpointed_equivalent_TFLOPS": 8 * params * 440 / checks["forward_backward_seconds"] / 1e12})
    work = flop_budget(config, config["workload"]["reference_sequence_length"], window_aware=False)
    delivery = [min(row["checkpointed_equivalent_TFLOPS"] for row in samples), max(row["checkpointed_equivalent_TFLOPS"] for row in samples)]
    fixed_seconds = (checks["gradient_clip_seconds"] + checks["optimizer_seconds"]) * config["model"]["planning_total_parameters"] / params
    positions_per_update = config["workload"]["m2_positions_per_optimizer_update"]
    rates = [1 / (work["total_flops_per_position"] / (tf * 1e12) + fixed_seconds / positions_per_update) for tf in delivery]
    updates = sustained["process_updates"]
    if not (sustained["changed_tensors"] == sustained["trainable_tensors"] == 381 and len(updates) == 12):
        raise ValueError("Unexpected sustained full-scope evidence")
    if abs(sum(row["total_seconds"] for row in updates[2:]) - steady["seconds"]) > 1e-8:
        raise ValueError("Sustained timing summary does not reproduce")
    return {"status": config["m2"]["status"], "executable_hours": None,
            "conditional_compute_only": True, "measured_minicpm_steady_updates_per_second": steady["updates_per_second"],
            "measured_minicpm_steady_processed_tokens_per_second": steady["processed_tokens_per_second"],
            "measured_minicpm_steady_phase_seconds": checks, "forward_backward_samples": samples,
            "conditional_delivered_TFLOPS": delivery, "conditional_projected_clip_optimizer_seconds_per_update": fixed_seconds,
            "conditional_projected_processed_positions_per_second": rates, "projection_work": work,
            "limits": ["The eight-parameter FLOP coefficient is a checkpointed-work convention, not measured silicon FLOPS.",
                       "Linear clipping/optimizer scaling and 2048-position efficiency extrapolation are unvalidated.",
                       "Unified host offload creates no additional physical memory; no executable full-scope MPS configuration is asserted."]}


def estimate(config_path=DEFAULT_CONFIG):
    config = read_json(config_path)
    folder = ROOT / config["evidence_directory"]
    manifest = verify_inputs(folder)
    inputs = folder / "inputs"
    residency = read_json(inputs / "residency_v2.json")
    intrinsic = read_json(inputs / "intrinsic_costs.json")
    model = config["model"]
    if residency["parameter_inventory"]["total_parameters"] != model["current_declared_integrated_floor_parameters"]:
        raise ValueError("Integrated inventory changed")
    if model["backbone_parameters"] != residency["parameter_inventory"]["backbone_parameters"]:
        raise ValueError("Backbone inventory changed")
    for key in ("provisional_position_mix", "illustrative_main_schedule_stage_shares"):
        if abs(sum(config[key].values()) - 1) > 1e-12:
            raise ValueError(f"Shares must sum to one: {key}")
    work = flop_budget(config, config["workload"]["reference_sequence_length"])
    m2 = m2_calibration(config, inputs)
    # The v2 floor contains two reader sites; replace those matrices with the
    # nine-site candidate, then add separately inventoried knowledge paths.
    partial_parameters = (model["current_declared_integrated_floor_parameters"] - 10485760
                          + intrinsic["fixed_parameters"]["memory_reader_matrices"]
                          + intrinsic["fixed_parameters"]["P_input_output_projections"]
                          + intrinsic["dynamic_state"]["P_parameters"])
    results = []
    for positions in config["user_selected_total_processed_positions"]:
        hardware = {}
        for device in config["hardware"]:
            device_work = flop_budget(config, config["workload"]["reference_sequence_length"],
                                      device.get("window_aware_attention", True))
            rates = [tf * 1e12 / device_work["total_flops_per_position"] for tf in device["delivered_reference_equivalent_TFLOPS"]]
            timing = hours_for_positions(positions, rates, device["gpus"], config["elapsed_overhead_factor"])
            campaign = [timing["gpu_hours"][i] * config["campaign_full_schedule_equivalents"] + device["qualification_gpu_hours"][i] for i in (0, 1)]
            price = device["price_usd_per_gpu_hour"]
            hardware[device["id"]] = {"gpus_per_job": device["gpus"], "status": device["status"],
                "modeled_flops_per_position": device_work["total_flops_per_position"],
                "assumed_processed_positions_per_second_entire_job": rates, "one_schedule": timing,
                "three_equivalent_campaign_gpu_hours_including_qualification": campaign,
                "one_schedule_usd_compute_only": [h * price for h in timing["gpu_hours"]] if price is not None else None,
                "campaign_usd_compute_only": [h * price for h in campaign] if price is not None else None}
        conditional_m2 = hours_for_positions(positions, m2["conditional_projected_processed_positions_per_second"], 1, config["elapsed_overhead_factor"])
        conditional_m2["hours_are_device_hours_not_cuda_gpu_hours"] = True
        conditional_m2["not_an_executable_budget"] = True
        conditional_m2["wall_days_at_half_time_availability"] = [h / 24 / config["m2"]["availability_fraction_for_calendar_example"] for h in conditional_m2["wall_hours"]]
        # Example of separate tasks: two complete candidates on the 3090 pair,
        # one complete candidate on the TITAN pair. No cross-pair synchronization.
        a = hardware["dual_3090ti"]["one_schedule"]["wall_hours"]
        b = hardware["dual_titan"]["one_schedule"]["wall_hours"]
        local = {"allocation": "Two whole schedules on 3090 Ti pair; one on TITAN pair, independently",
                 "wall_hours_excluding_one_time_qualification": [max(2*a[i], b[i]) for i in (0, 1)],
                 "aggregate_gpu_hours_excluding_one_time_qualification": [2 * (2*a[i] + b[i]) for i in (0, 1)],
                 "add_pair_qualification_gpu_hours": [32, 80],
                 "not_one_96_GB_training_job": True}
        results.append({"total_charged_positions": positions, "main_schedule_position_split": {k: round(v*positions) for k,v in config["illustrative_main_schedule_stage_shares"].items()},
                        "hardware": hardware, "m2_conditional_compute_only": conditional_m2, "independent_local_campaign_example": local})
    return {"format": "replica.sparkx_formal_training_budget_estimate.v1", "kind": "conditional_planning_ranges_not_benchmarks",
            "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path),
            "calculator_sha256": sha256(Path(__file__)), "input_manifest_sha256": sha256(folder / "input_manifest.json"),
            "snapshot_source_commit": manifest["source_commit"], "work_budget": work,
            "context_sensitivity": [flop_budget(config, length, aware) for length,aware in [(2048,True),(8192,True),(2048,False),(8192,False)]],
            "integrated_parameter_account": {"versioned_floor": model["current_declared_integrated_floor_parameters"],
                "partial_nine_reader_plus_P_inventory": partial_parameters, "planning_total_not_measured": model["planning_total_parameters"],
                "remaining_parameter_reserve_not_inventory": model["planning_total_parameters"] - partial_parameters},
            "parameter_state_GiB_at_planning_total": {str(rate)+"_bytes_per_parameter": model["planning_total_parameters"]*rate/1024**3 for rate in (12,14,16,18)},
            "source_residency_projection_not_measured": residency["projected_full_scope_training"],
            "m2_calibration": m2, "scenarios": results,
            "limitations": ["No new model, GPU, cloud, external media or paid resource execution.",
                            "Only MiniCPM timing is measured; every Spark and CUDA hour is a conditional planning estimate.",
                            "Rate bands and 25-50% overhead are assumptions, not statistical confidence intervals.",
                            "Three full-schedule equivalents are a finite campaign reserve, not a guaranteed product-quality budget.",
                            "Full trainable scope is preserved; changing numerical precision, offload, optimizer implementation or backend requires separate qualification.",
                            "Specified exclusions and any additional experiments must be added to the budget."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior evidence; choose a fresh output")
    result = estimate(args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for scenario in result["scenarios"]:
        print(json.dumps({"charged_positions": scenario["total_charged_positions"],
                          "H200_one_schedule_gpu_hours": scenario["hardware"]["do_h200"]["one_schedule"]["gpu_hours"],
                          "H200_campaign_gpu_hours": scenario["hardware"]["do_h200"]["three_equivalent_campaign_gpu_hours_including_qualification"]}))


if __name__ == "__main__":
    main()
