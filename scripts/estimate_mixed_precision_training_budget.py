#!/usr/bin/env python3
"""Offline, phase-explicit planning; no torch, devices, payloads or networking."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs/mixed_precision_training_budget_v1.json"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(folder):
    manifest = read(folder / "input_manifest.json")
    result = {}
    for item in manifest["inputs"]:
        path = ROOT / item["path"]
        if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Evidence changed: {path}")
        result[path.name] = path
    return result


def inventory(rows, backbone, steps):
    hidden, auxiliary, shapes = 0, 0, []
    for row in rows:
        name, shape = row["name"], row["shape"]
        n = math.prod(shape)
        is_hidden = len(shape) == 2 and not any(
            token in name for token in ("embed", "lm_head", ".g_proj."))
        if is_hidden:
            hidden += n
            r, c = sorted(shape)
            shapes.append({"name": name, "shape": shape,
                           "ns_flops": steps * (4 * r * r * c + 2 * r ** 3)})
        else:
            auxiliary += n
    return {"backbone": backbone, "parameters": hidden + auxiliary,
            "hidden_muon_parameters": hidden, "auxiliary_parameters": auxiliary,
            "hidden_matrices": len(shapes), "ns_flops_per_update": sum(x["ns_flops"] for x in shapes),
            "routing": "Hidden backbone matrices only; g_proj and all other gates/norms/embeddings/heads use auxiliary AdamW."}


def work(inv, cfg, model, length, dense=False):
    if model["backbone"] == "minicpm":
        spans, heads, head_dim = 42 * length, 16, 128
    else:
        spans, heads, head_dim = (36 * length if dense else 9 * length + 27 * min(length, 512)), 16, 256
    return {"checkpointed_parameter_flops_per_position": 8 * inv["parameters"],
            "attention_flops_per_position": 16 * heads * head_dim * spans,
            "extension_reserve_per_position": model["extension_flops_per_position"]}


def memory(inv, total):
    h, a = inv["hidden_muon_parameters"], total - inv["hidden_muon_parameters"]
    # Separate copies explicitly. These are lower persistent-state accounts.
    profiles = {
        "all_fp32_adamw": 16 * total,
        "bf16_adamw_fp32_master_bf16_gradient": 16 * total,
        "bf16_adamw_fp32_master_fp32_gradient": 18 * total,
        "bf16_muon_fp32_master_and_momentum": 12 * h + 16 * a,
        "bf16_muon_fp32_master_momentum_and_gradient": 14 * h + 18 * a,
        "bf16_muon_fp32_master_bf16_momentum": 10 * h + 16 * a,
        "bf16_muon_no_master_fp32_momentum": 8 * h + 12 * a,
        "fp32_muon8bit_hypothesis": (9 + 4 / 2048) * h + 16 * a,
    }
    return {"hidden_muon_parameters": h, "auxiliary_adamw_parameters": a,
            "persistent_GiB": {k: v / 2**30 for k, v in profiles.items()},
            "excludes": "Activations, KV/state histories, gathered matrices, casts, logits, optimizer scratch, allocator, checkpoint/reload duplication and host OS; no fit certification."}


def hours(positions, step_seconds, batch, gpus, allowance):
    if positions <= 0 or batch <= 0 or not 0 < step_seconds[0] <= step_seconds[1] or gpus < 1:
        raise ValueError("Positive work, ordered times and GPU count required")
    wall = [positions / batch * step_seconds[i] / 3600 * allowance[i] for i in range(2)]
    return {"wall_hours": wall, "gpu_hours": [x * gpus for x in wall],
            "continuous_wall_days": [x / 24 for x in wall]}


def schedule(config, seconds, gpus, qualification, price=None):
    result = []
    for n in config["positions"]:
        row = {"positions": n, **hours(n, seconds, config["positions_per_update"], gpus, config["elapsed_allowance"])}
        row["campaign_gpu_hours"] = [config["campaign_schedule_equivalents"] * row["gpu_hours"][i] + qualification[i] for i in range(2)]
        if price is not None:
            row["gpu_compute_USD"] = [x * price for x in row["gpu_hours"]]
            row["campaign_gpu_compute_USD"] = [x * price for x in row["campaign_gpu_hours"]]
        result.append(row)
    return result


def cuda_phase(config, model, inv, hardware, length=None, optimizer="muon"):
    length = length or config["sequence_length"]
    dense = hardware.get("spark_dense_attention", False)
    w = work(inv, config, model, length, dense)
    flops = sum(w.values())
    delivery = hardware.get("spark_model_TFLOPS", hardware["model_TFLOPS"]) if model["backbone"] == "spark" else hardware["model_TFLOPS"]
    ratio = config["unlock"]["reported_bf16_after_TFLOPS"] / config["unlock"]["reported_bf16_before_TFLOPS"]
    phases = []
    for endpoint in range(2):  # Fast then slow; rate arrays are slow then fast.
        index = 1 - endpoint
        model_sec = flops * config["positions_per_update"] / (delivery[index] * 1e12)
        muon_sec = inv["ns_flops_per_update"] / (hardware["muon_TFLOPS"][index] * 1e12) if optimizer == "muon" else 0
        factor = 1.0
        if hardware.get("unlock"):
            fraction = config["unlock"]["accelerable_model_time_fraction"][index]
            factor = 1 - fraction + fraction / ratio
            model_sec *= factor
            muon_sec /= ratio
        other = hardware["other_seconds_per_update_at_4_4B"][endpoint] * model["planning_parameters"] / 4.4e9
        phases.append({"model_seconds": model_sec, "muon_seconds": muon_sec,
                       "other_seconds": other, "total_seconds": model_sec + muon_sec + other,
                       "unlock_model_phase_factor": factor})
    seconds = [p["total_seconds"] for p in phases]
    return {"work": w, "sequence_length": length, "phases_fast_slow": phases,
            "seconds_per_update_fast_slow": seconds,
            "processed_positions_per_second_slow_fast": [config["positions_per_update"] / x for x in seconds[::-1]],
            "updates_per_second_slow_fast": [1 / x for x in seconds[::-1]]}


def estimate(config_path=DEFAULT):
    cfg = read(config_path)
    folder = ROOT / cfg["evidence_directory"]
    evidence = inputs(folder)
    residency = read(evidence["residency_v2.json"])
    moved_ids = {"vision_encoder", "audio_encoder", "vision_projector", "audio_projector"}
    moved_parameters = sum(c["parameters"] for c in residency["parameter_inventory"]["components"] if c["id"] in moved_ids)
    if moved_parameters != cfg["pipeline"]["encoder_and_bridge_parameters"]:
        raise ValueError("Pipeline placement differs from the pinned component inventory")
    mini = read(evidence["minicpm_tensor_audit.json"])
    inv = {
        "minicpm": inventory([{"name": k, "shape": v["shape"]} for k, v in mini["tensors"].items()], "minicpm", cfg["optimizer"]["ns_steps"]),
        "spark": inventory(read(evidence["spark_tensor_index.json"])["tensors"], "spark", cfg["optimizer"]["ns_steps"]),
    }
    mini_cfg, spark_cfg = read(evidence["minicpm_config.json"]), read(evidence["spark_config.json"])
    for c, expected in ((mini_cfg, (42, 16, 128)), (spark_cfg, (36, 16, 256))):
        if (c["num_hidden_layers"], c["num_attention_heads"], c["head_dim"]) != expected:
            raise ValueError("Attention geometry differs from reviewed arithmetic")
    if spark_cfg["layer_types"].count("full_attention") != 9 or spark_cfg["sliding_window"] != 512:
        raise ValueError("Spark attention schedule changed")
    cached = read(evidence["minicpm_cached_analysis.json"])
    sustained = read(evidence["fused_sustained_report.json"])
    native = read(evidence["native_summary.json"])
    if inv["minicpm"]["ns_flops_per_update"] != cached["muon_arithmetic"]["ns_flops_per_update"]:
        raise ValueError("MiniCPM Muon arithmetic mismatch")
    if inv["minicpm"]["parameters"] != sustained["trainable_parameters"]:
        raise ValueError("MiniCPM measured scope mismatch")
    samples = []
    for key in ("full_55", "full_256", "full_512"):
        case = cached["cached_cases"][key]
        length = case["actual_sequence_length"]["maximum"]
        if length != case["actual_sequence_length"]["minimum"]:
            raise ValueError("Uniform calibration sequences required")
        flops = sum(work(inv["minicpm"], cfg, cfg["models"][0], length).values())
        samples.append({"case": key, "positions": 8 * length,
                        "forward_backward_seconds": case["mean_forward_backward_seconds"],
                        "equivalent_TFLOPS": flops * 8 * length / case["mean_forward_backward_seconds"] / 1e12})
    m2_delivery = [min(x["equivalent_TFLOPS"] for x in samples), max(x["equivalent_TFLOPS"] for x in samples)]
    clip = sustained["performance"]["steady"]["mean_seconds_by_phase"]["gradient_clip_seconds"]
    measured_muon = native["measured_arms"]["muon_reference"]
    muon_delivery = [inv["minicpm"]["ns_flops_per_update"] / seconds / 1e12 for seconds in (
        measured_muon["stage_mean_seconds"]["optimizer_seconds"], measured_muon["updates"][-1]["optimizer_seconds"])]
    output = {"format": "replica.mixed_precision_training_budget.v1", "kind": cfg["kind"],
              "config_sha256": digest(config_path), "calculator_sha256": digest(Path(__file__)),
              "input_manifest_sha256": digest(folder / "input_manifest.json"),
              "backbone_inventories": inv, "m2_measured_calibration": {"samples": samples,
              "model_equivalent_TFLOPS": m2_delivery, "muon_optimizer_equivalent_TFLOPS": muon_delivery,
              "steady_fused": sustained["performance"]["steady"]}, "models": []}
    hardware = {x["id"]: x for x in cfg["hardware"]}
    for model in cfg["models"]:
        b = inv[model["backbone"]]
        result = {"model": model, "memory": memory(b, model["planning_parameters"]), "hardware": []}
        for h in hardware.values():
            phase = cuda_phase(cfg, model, b, h)
            result["hardware"].append({"id": h["id"], "label": h["label"], "measured": False,
                "gpus": h["gpus"], **phase,
                "scenarios": schedule(cfg, phase["seconds_per_update_fast_slow"], h["gpus"], h["qualification_gpu_hours"], h["price_per_gpu_hour"]),
                "length_8192_sensitivity": cuda_phase(cfg, model, b, h, 8192)})
        adam = cuda_phase(cfg, model, b, hardware["h200"], optimizer="adamw")
        result["h200_fused_adamw_same_phase_assumptions"] = {**adam,
            "limitation": "Removes Muon GEMMs only, keeps other allowance; actual AdamW memory/step overhead differs. H200 has residency headroom; not a quality-equivalence assertion.",
            "scenarios": schedule(cfg, adam["seconds_per_update_fast_slow"], 1, [8, 16], 4.47)}
        titan_fp32 = dict(hardware["titan"], muon_TFLOPS=[4, 12])
        fallback = cuda_phase(cfg, model, b, titan_fp32)
        result["titan_fp32_muon_fallback"] = {**fallback,
            "scenarios": schedule(cfg, fallback["seconds_per_update_fast_slow"], 2, [16, 40])}
        previous = dict(hardware["rtx3090ti_unlocked"], model_TFLOPS=cfg["rtx3090ti_previous_delivery_fallback_TFLOPS"])
        fallback = cuda_phase(cfg, model, b, previous)
        result["unlocked_3090ti_previous_delivery_fallback"] = {**fallback,
            "scenarios": schedule(cfg, fallback["seconds_per_update_fast_slow"], 2, [24, 56])}
        m2_work = sum(work(b, cfg, model, cfg["sequence_length"], dense=True).values())
        m2_seconds = []
        for endpoint in range(2):
            i = 1 - endpoint
            speed = cfg["m2"]["bf16_GEMM_speed_sensitivity"][i]
            model_sec = m2_work * cfg["positions_per_update"] / (m2_delivery[i] * speed * 1e12)
            optimizer_sec = b["ns_flops_per_update"] / (muon_delivery[i] * speed * 1e12)
            clip_sec = clip * model["planning_parameters"] / inv["minicpm"]["parameters"]
            m2_seconds.append(model_sec + optimizer_sec + clip_sec)
        result["m2_bf16_muon_what_if"] = {"measured": False, "execution_qualified": False,
            "executable_budget": None, "seconds_per_update_fast_slow": m2_seconds,
            "positions_per_second_slow_fast": [cfg["positions_per_update"] / x for x in m2_seconds[::-1]],
            "conditional_scenarios": schedule(cfg, m2_seconds, 1, cfg["m2"]["qualification_device_hours"])}
        if model["full_architecture_projection"]:
            pipe = cfg["pipeline"]
            moved = pipe["encoder_and_bridge_parameters"]
            backbone_model = dict(model, planning_parameters=model["planning_parameters"] - moved,
                                  extension_flops_per_position=model["extension_flops_per_position"] - pipe["encoder_flops_per_processed_position"])
            backbone_phase = cuda_phase(cfg, backbone_model, b, hardware[pipe["backbone_hardware"]])
            seconds = []
            for j, phase in enumerate(backbone_phase["phases_fast_slow"]):
                encoder = pipe["encoder_flops_per_processed_position"] * cfg["positions_per_update"] / (pipe["encoder_TFLOPS"][1-j] * 1e12)
                seconds.append(max(phase["model_seconds"], encoder) * pipe["bubble_and_transfer_factor"][j]
                               + phase["muon_seconds"] + max(phase["other_seconds"], pipe["encoder_optimizer_seconds"][j]))
            result["four_gpu_joint_pipeline"] = {"measured": False, "implemented": False,
                "seconds_per_update_fast_slow": seconds, "backbone_pair_memory": memory(b, backbone_model["planning_parameters"]),
                "encoder_pair_persistent_GiB_fp16_master_adamw": moved * 16 / 2**30,
                "scenarios": schedule(cfg, seconds, 4, pipe["qualification_gpu_hours"])}
        output["models"].append(result)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = estimate(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"Wrote analytical estimate: {args.output}")


if __name__ == "__main__":
    main()
