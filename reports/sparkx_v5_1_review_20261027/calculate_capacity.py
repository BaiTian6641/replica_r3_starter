"""Analytical payload/MAC ledger only. Does not import or execute model code."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIB = 1024**2


def inventory(c):
    inv = c["residency_inventory"]
    components = {}
    for name, value in inv["components"].items():
        components[name] = {
            "parameter_elements": value["parameter_elements"],
            "bytes_per_element": value["bytes_per_element"],
            "mib": value["parameter_elements"] * value["bytes_per_element"] / MIB,
            "precision": value["precision"],
            "evidence_class": value["evidence_class"],
            "measured_fit": False,
            "full_training_qualified": False,
        }
    return {
        "kind": "analytical_partial_residency_not_measured_fit_or_training_admission",
        "measured_fit": False,
        "full_training_qualified": False,
        "components": components,
        "unsealed": {
            name: value["evidence_class"]
            for name, value in inv["unsealed_components"].items()
        },
        "restart_transients": inv["restart_transients"],
    }


def calculate():
    c = json.loads((ROOT / "configs/sparkx_2_6_v5_1_direction_design.json").read_text())
    p, e, m = c["plastic"], c["E2"], c["Mkv"]
    n = p["matrices"] * p["input_width"] * p["hidden_width"]
    compact = (
        (e["address_keys"] * e["key_width"] + e["semantic_width"]
         + e["visual_compact"][0] * e["visual_compact"][1]
         + e["audio_compact"][0] * e["audio_compact"][1]) * e["feature_bytes"]
        + e["metadata_bytes"] + e["lexical_hashes"] * e["hash_bytes"]
        + e["entity_refs"] * e["entity_ref_bytes"]
        + e["literal_ids"] * e["literal_id_bytes"]
    )
    hot = sum(e[k][0] * e[k][1] for k in ("hot_visual", "hot_audio")) * e["feature_bytes"]
    entity = e["entity_modal_keys"][0] * e["entity_modal_keys"][1] * e["feature_bytes"] + e["entity_metadata_bytes"]
    profiles = {}
    for name, v in p["profiles"].items():
        params = v["experts"] * n
        ep = v["records"] * compact + v["hot_records"] * hot + v["entities"] * entity
        profiles[name] = {
            "P_parameters": params,
            "P_weight_mib": params * 4 / MIB,
            "P_two_moment_mib": params * 8 / MIB,
            "P_live_mib": params * 12 / MIB,
            "E2_payload_mib": ep / MIB,
            "two_global_dense_query_MAC": e["max_global_dense_queries_per_decision"] * v["records"] * e["address_keys"] * e["key_width"],
            "P_plus_E2_payload_mib": (params * 12 + ep) / MIB,
            "selected_two_expert_weight_moment_gradient_staging_mib": n * 2 * 16 / MIB,
            "mutable_envelope_mib": v["mutable_envelope_mib"],
            "separate_replay_envelope_mib": v["replay_envelope_mib"],
            "measured_fit": False,
        }
    packets = sum(e[k] for k in ("compact_read_packets", "hot_visual_read_packets", "hot_audio_read_packets", "workspace_read_packets"))
    mv = m["rows"] * m["value_width"]
    return {
        "kind": "analytical_partial_payload_not_latency_or_fit",
        "expert_parameters": n,
        "record_bytes": {"compact": compact, "hot": hot, "entity": entity},
        "profiles": profiles,
        "wide_P16_parameters": 16 * 3 * 512 * 2048,
        "P_expert_matrix_MAC_per_position_at_9_sites": {str(a): a * 9 * n for a in (2, 4, 16, 64)},
        "reader_packets_max": packets,
        "projected_reader_KV_mib": {str(s): packets * e["packet_width"] * 2 * 2 * s / MIB for s in (7, 9)},
        "Mkv": {
            "value_parameters": mv, "value_weight_mib": mv * 4 / MIB,
            "value_two_moment_mib": mv * 8 / MIB,
            "value_live_mib": mv * 12 / MIB,
            "fixed_product_key_parameters": m["subkey_tables"] * m["subkeys_per_table"] * m["subkey_width"],
            "fixed_product_key_FP32_mib": m["subkey_tables"] * m["subkeys_per_table"] * m["subkey_width"] * 4 / MIB,
            "dense_gradient_fallback_mib": mv * 4 / MIB,
            "subkey_score_MAC_per_query": m["subkey_tables"] * m["subkeys_per_table"] * m["subkey_width"],
            "joint_candidates": m["read_top_k"]**2,
            "selected_row_weight_moment_gradient_staging_mib": m["max_unique_updated_rows_per_transaction"] * m["value_width"] * 16 / MIB,
        },
        "Mkv_growth": {
            str(rows): {"value_parameters": rows * m["value_width"],
                        "FP32_values_two_moments_mib": rows * m["value_width"] * 12 / MIB,
                        "subkeys_per_half": math.isqrt(rows),
                        "subkey_score_MAC_per_query": 2 * math.isqrt(rows) * m["subkey_width"]}
            for rows in m["later_row_capacity_controls"]
        },
        "nonlinear_F_candidate": {"parameters": 3 * 8 * 4 * 2 * 128 * 256, "FP32_weight_mib": 24, "same_size_velocity_mib": 24},
        "P64_rate_scalars": 64 * 3,
        "P64_full_parameterwise_rate_FP32_mib": 64 * n * 4 / MIB,
        "AV_16s_example_positions": {"coarse": 4 * 256 + 16 * 12.5 + 64 + 32, "two_extra_crops": 4 * 256 + 2 * 256 + 16 * 12.5 + 64 + 32},
        "residency_inventory": inventory(c),
        "exclusions": ["unsealed stable encoders and heads", "decoder KV", "activations", "teacher snapshots/logits", "variable literal/media payloads", "indices and journals", "allocator/kernel temporaries", "full candidate copies beyond selected staging", "measured sparse-kernel bandwidth"],
    }


if __name__ == "__main__":
    target = Path(__file__).with_name("analytical_capacity.json")
    target.write_text(json.dumps(calculate(), indent=2) + "\n")
    print(target.relative_to(ROOT))
