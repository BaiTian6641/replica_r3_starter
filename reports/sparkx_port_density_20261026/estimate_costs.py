#!/usr/bin/env python3
"""Analytical port-density counts. No model/framework imports or device access."""

import json
from pathlib import Path


def compute():
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "configs/sparkx_2_6_4b_port_density_study_v1.json").read_text())
    baseline = json.loads((root / plan["baseline_config"]).read_text())
    result = {"format": "sparkx.port_density.analytical_costs.v1", "scope": "analytical_not_runtime_measured", "models": {}}
    for key, prefix in [("sparkx", "spark"), ("minicpm5", "minicpm")]:
        model = baseline["models"][key]
        d, layers = model["hidden_size"], model["layers"]
        fused = d * 128 + 512 * d
        factored = d * 128 + 512 * 64 + 64 * d
        # Each existing reader owns Q/O, K/V, two affine LayerNorms and a scalar gate.
        current_pair = 2 * (2 * d * 256 + 2 * 256 ** 2 + 2 * d + 2 * 256 + 1)
        rows = []
        for arm in plan["arms"]:
            n = len(arm[prefix + "_read_layers"])
            p = n * fused
            rows.append({
                "arm": arm["name"], "read_ports": n,
                "fused_query_output_parameters": p,
                "fused_BF16_weight_MiB": p * 2 / 2 ** 20,
                "fused_FP32_weight_gradient_Adam_MiB": p * 16 / 2 ** 20,
                "fused_query_output_MACs_per_action_position": p,
                "factor64_query_output_parameters": n * factored,
                "factor64_BF16_weight_MiB": n * factored * 2 / 2 ** 20,
                "current_memory_workspace_pair_parameters": n * current_pair,
                "current_pair_BF16_weight_MiB": n * current_pair * 2 / 2 ** 20,
            })
        result["models"][key] = {
            "width": d, "layers": layers, "rows": rows,
            "fused_parameters_per_port": fused,
            "factor64_parameters_per_port": factored,
            "current_memory_workspace_pair_parameters_per_port": current_pair,
            "three_shared_full_rank_projection_sets_parameters": 3 * fused,
            "three_shared_full_rank_projection_sets_BF16_MiB": 3 * fused * 2 / 2 ** 20,
            "all_layer_fused_parameter_fraction_of_backbone": layers * fused / model["unique_parameters"],
            "one_BF16_8192_position_hidden_activation_MiB": 8192 * d * 2 / 2 ** 20,
            "one_such_activation_per_layer_MiB_illustration_only": layers * 8192 * d * 2 / 2 ** 20,
        }
    result["state"] = {
        "shared_spark_bank_tensor_MiB": 4096 * 896 * 2 / 2 ** 20,
        "duplicated_bank_at_all_36_layers_MiB": 36 * 4096 * 896 * 2 / 2 ** 20,
        "shared_fast_pair_MiB": 2 * 256 * 128 * 4 / 2 ** 20,
        "independent_fast_pair_at_all_36_layers_MiB": 36 * 2 * 256 * 128 * 4 / 2 ** 20,
        "full_bank_scan_MACs_two_queries": 2 * 4096 * 128,
        "full_bank_scan_MACs_one_query_at_each_36_layers_rejected_default": 36 * 4096 * 128,
    }
    result["exclusions_and_limits"] = [
        "Fused/factored counts exclude norms, gates, fusion/key/value/selection/writer/sensory modules.",
        "Existing reader-pair counts include its norms/gates, but exclude sensing, writers and fast-memory branches.",
        "Projection MACs exclude attention, bank scans, nonlinearities, normalization and actual kernel costs.",
        "Shared weights still execute at every active reader; sharing is not arithmetic skipping.",
        "16 bytes per parameter assumes FP32 weight, gradient and two Adam moments only.",
        "Activation examples are illustrative tensors, not a saved-tensor or peak-memory model.",
        "No latency, training-memory fit, quality or energy result is established.",
    ]
    return result


if __name__ == "__main__":
    costs = compute()
    (Path(__file__).resolve().parent / "analytical_costs.json").write_text(json.dumps(costs, indent=2) + "\n")
    print(json.dumps(costs["models"]["sparkx"]["rows"], indent=2))
