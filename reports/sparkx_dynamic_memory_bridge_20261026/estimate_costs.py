"""Analytical proposal accounting only. Standard library; no model imports/runs."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/sparkx_2_6_4b_dynamic_memory_bridge_study_v1.json"
MIB = 1024 ** 2


def calculate(config):
    ports = []
    for model, geometry in config["geometry"].items():
        hidden = geometry["hidden_size"]
        for width in config["ports"]["payload_width_controls"]:
            per_site = 2 * hidden * width + 2 * width * width
            for sites in (2, 6, 9, geometry["layers"]):
                weights = sites * per_site
                ports.append({
                    "model": model, "sites": sites, "width": width,
                    "matrix_parameters": weights,
                    "BF16_weight_MiB": 2 * weights / MIB,
                    "FP32_weight_gradient_Adam2_MiB": 16 * weights / MIB,
                    "read_MACs_per_position_with_16_cached_KV_packets":
                        sites * (2 * hidden * width + 2 * 16 * width),
                    "KV_projection_MACs_per_16_packet_refresh":
                        sites * 2 * 16 * width * width,
                })

    fast = config["memory"]["fast_association"]
    fast_states = []
    for experts in fast["capacity_candidates"]:
        elements = (fast["depth_groups"] * experts * fast["heads"]
                    * fast["key_width"] * fast["value_width"])
        fast_states.append({
            "groups": fast["depth_groups"], "experts_per_group": experts,
            "FP32_state_elements": elements, "FP32_state_MiB": 4 * elements / MIB,
            "old_plus_candidate_state_MiB": 8 * elements / MIB,
            "optional_BF16_meta_learned_initial_weights_MiB": 2 * elements / MIB,
            "dense_matrix_read_MACs_one_group_one_query":
                experts * fast["heads"] * fast["key_width"] * fast["value_width"],
            "top2_matrix_read_MACs_one_group_one_query":
                min(2, experts) * fast["heads"] * fast["key_width"] * fast["value_width"],
        })
    address = config["memory"]["fast_addressing"]
    address_elements = address["depth_groups"] * address["width"] ** 2

    banks = []
    for records, shards in ((1024, 32), (4096, 64), (16384, 128), (65536, 256)):
        per_shard = records // shards
        banks.append({
            "records": records, "tensor_MiB": records * 896 * 2 / MIB,
            "metadata_cap_MiB": records * 512 / MIB,
            "bank_payload_plus_metadata_MiB": records * (896 * 2 + 512) / MIB,
            "flat_two_query_dot_MACs": 2 * records * 128,
            "balanced_shard_count": shards, "selected_shards": 2,
            "balanced_hierarchical_two_query_dot_MACs":
                2 * (shards + 2 * per_shard) * 128,
            "bank_alone_fits_128MiB": records * (896 * 2 + 512) <= 128 * MIB,
        })

    large = config["memory"]["large_attention_memory_candidate"]
    hidden = config["geometry"]["spark"]["hidden_size"]
    heads, dk, dv, sites = (large[k] for k in ("heads", "key_width", "value_width", "read_sites"))
    # Q and K map hidden -> heads*dk; V maps hidden -> heads*dv; O reverses V.
    full_projections = sites * (2 * hidden * heads * dk + 2 * hidden * heads * dv)
    large_state_elements = sites * heads * dk * dv
    experts = []
    for count in (16, 64, 128):
        per_expert = 3 * 512 * 2048  # bias-free gated MLP: up, gate, down
        experts.append({
            "experts": count, "matrix_parameters": count * per_expert,
            "BF16_weight_MiB": 2 * count * per_expert / MIB,
            "top2_active_matrix_MACs_per_processed_vector": 2 * per_expert,
            "dense_matrix_MACs_per_processed_vector": count * per_expert,
        })

    return {
        "format": "sparkx.dynamic_memory_bridge.analytical_costs.v1",
        "date_local": "2026-10-26", "runtime_measured": False,
        "units": {"MiB": MIB, "BF16_bytes": 2, "FP32_bytes": 4,
                  "MAC": "one multiply-accumulate; not a measured latency"},
        "port_matrices": ports, "fast_expert_states": fast_states,
        "fast_addressing": {"elements": address_elements,
                            "FP32_MiB": address_elements * 4 / MIB},
        "bank_and_search_examples": banks,
        "large_memory_candidate": {
            "sites": sites, "heads": heads, "dk": dk, "dv": dv,
            "FP32_state_MiB": large_state_elements * 4 / MIB,
            "separate_full_QKVO_matrix_parameters": full_projections,
            "BF16_projection_MiB": full_projections * 2 / MIB,
            "FP32_weight_gradient_Adam2_MiB": full_projections * 16 / MIB,
        },
        "conditional_full_rank_skill_pool_examples": experts,
        "exclusions_and_limits": [
            "Port matrix counts exclude norms, gates, encoders, core, routers and other modules.",
            "Read MAC example assumes 16 already projected packets, not 16 arbitrarily large records.",
            "Independent K/V projections are refreshed per site; reuse requires a valid version/state key.",
            "FP32 Adam example counts weight, gradient and two moments; no activations/workspaces.",
            "Fast state is not durable model learning; learned initialization weights are listed separately.",
            "Old+candidate states exclude updater gradients, momentum, replay and allocator overhead.",
            "Sparse counts require actual dispatch; routing, gathers, index maintenance and paging are excluded.",
            "Hierarchical examples assume balanced shards and do not guarantee exact retrieval.",
            "65536 paired records already exceed the 128MiB profile before fast/core state.",
            "Expert read counts omit query projections, gating, nonlinearities and module scheduling.",
            "None of these arithmetic examples qualifies device fit, real-time performance or quality.",
        ],
    }


if __name__ == "__main__":
    result = calculate(json.loads(CONFIG.read_text()))
    target = Path(__file__).with_name("analytical_costs.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {target.relative_to(ROOT)}; analytical only")
