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
            "tensor_count": value.get("tensor_count"),
            "trainable_in_offline_full_finetuning":
                value.get("trainable_in_offline_full_finetuning", False),
            "missing_prerequisite": value.get("missing_prerequisite"),
            "sealed_source_manifest": value.get("sealed_source_manifest"),
            "sealed_source_sha256": value.get("sealed_source_sha256"),
            "measured_fit": False,
            "full_training_qualified": False,
        }
    return {
        "kind": "analytical_partial_residency_not_measured_fit_or_training_admission",
        "measured_fit": False,
        "full_training_qualified": False,
        "components": components,
        "sealed_source_inventory": inv["sealed_source_inventory"],
        "unsealed": {
            name: {
                "evidence_class": value["evidence_class"],
                "missing_prerequisite": value.get("missing_prerequisite"),
                "reason": value.get("reason"),
            }
            for name, value in inv["unsealed_components"].items()
        },
        "restart_transients": inv["restart_transients"],
        "training_contract": inv["training_contract"],
        "feasibility": inv["feasibility"],
    }


def activation_workloads(c):
    """Derive per-view activation arithmetic from declared patch scenarios."""
    declared = c["residency_inventory"]["activation_workloads"]
    routes = {}
    for name, value in declared.items():
        if not isinstance(value, dict):
            continue
        patches = value.get("patches", value.get("padded_patches"))
        merge = value.get("merge", [1, 1])
        outputs = value.get("outputs")
        width = value["width"]
        pre_merge = patches if merge != [1, 1] else None
        route = {
            "encoder": value["encoder"],
            "width": width,
            "patch_size": value.get("patch_size"),
            "image_size": value.get("image_size"),
            "patches": patches,
            "merge": merge,
            "outputs": outputs,
            "elements_per_view_per_layer": outputs * width,
            "bf16_bytes_per_view_per_layer": outputs * width * 2,
            "attention_pair_work_per_layer": outputs * outputs,
        }
        if pre_merge is not None:
            route["pre_merge_elements_per_view_per_layer"] = pre_merge * width
            route["pre_merge_bf16_bytes_per_view_per_layer"] = pre_merge * width * 2
        if "valid_patches" in value:
            route["valid_patches"] = value["valid_patches"]
            route["padded_patches"] = value["padded_patches"]
        if "expected_256" in value:
            route["expected_256"] = value["expected_256"]
        routes[name] = route
    return {
        "kind": declared["kind"],
        "measured": False,
        "routes": routes,
    }


def precision_bytes(c):
    """Reconcile storage bytes for each distinct precision identity."""
    rows = {}
    for name in ("official_bf16", "official_fp8_storage",
                 "dequantized_fp8_derivative", "native_fp8_execution"):
        reconciliation = c["precision_identities"][name].get("byte_reconciliation")
        if reconciliation is None:
            continue
        row = dict(reconciliation)
        payload = reconciliation.get("payload_bytes")
        if payload is not None:
            row["computed_payload_mib"] = payload / MIB
        row["reconciled"] = _precision_reconciled(name, reconciliation)
        rows[name] = row
    return rows


def _precision_reconciled(name, value):
    if name == "official_bf16":
        return (value["payload_bytes"]
                == value["unique_parameter_elements"] * 2
                and value["payload_mib"] == value["payload_bytes"] / MIB)
    if name == "official_fp8_storage":
        return (value["f8_e4m3_payload_bytes"]
                + value["bf16_payload_bytes"] == value["payload_bytes"]
                and value["quantized_parameter_elements"]
                + value["bf16_unquantized_elements"]
                == value["declared_unique_model_parameter_elements"]
                and value["scale_tensor_elements"]
                == value["serialized_element_elements"]
                - value["declared_unique_model_parameter_elements"]
                and value["payload_mib"] == value["payload_bytes"] / MIB)
    if name == "dequantized_fp8_derivative":
        return (value["dtype_payload_bytes"]["BF16"]
                == value["unique_parameter_elements"] * 2
                == value["payload_bytes"]
                and value["serialized_file_bytes"] >= value["payload_bytes"]
                and value["payload_bytes"] != value["source_fp8_payload_bytes"])
    if name == "native_fp8_execution":
        return value["residency_bytes"] is None
    return False


def dependency_order(c):
    """Resolve every requires entry against earlier nodes and declarations."""
    contract = c["dependency_contract"]
    available: set[str] = set()
    declared: dict[str, str] = {}
    nodes = []
    for node in c["learning_dependency_order"]:
        unresolved = [item for item in node["requires"] if item not in available]
        for item in node.get("declares", []) + node.get("provides", []):
            declared.setdefault(item, node["id"])
        nodes.append({
            "order": node["order"],
            "id": node["id"],
            "requires": node["requires"],
            "preconditions": node.get("preconditions", []),
            "constraints": node.get("constraints", []),
            "declares": node["declares"],
            "provides": node.get("provides", []),
            "unresolved_requires": unresolved,
        })
        available |= ({node["id"]} | set(node.get("declares", []))
                      | set(node.get("provides", [])))
    return {
        "format": contract["format"],
        "required_sequence": contract["required_sequence"],
        "sequence": [node["id"] for node in nodes],
        "nodes": nodes,
        "all_requires_resolved": all(not node["unresolved_requires"]
                                     for node in nodes),
        "declared_capabilities": dict(sorted(declared.items())),
        "native_gate_refs": contract["native_gate_refs"],
        "rs_gates": contract["rs_gates"],
        "gate_status": contract["gate_status"],
        "promotion": contract["promotion"],
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
        "activation_workloads": activation_workloads(c),
        "precision_byte_reconciliation": precision_bytes(c),
        "dependency_order": dependency_order(c),
        "residency_inventory": inventory(c),
        "exclusions": ["unsealed stable encoders and heads", "decoder KV", "activations", "teacher snapshots/logits", "variable literal/media payloads", "indices and journals", "allocator/kernel temporaries", "full candidate copies beyond selected staging", "measured sparse-kernel bandwidth"],
    }


if __name__ == "__main__":
    target = Path(__file__).with_name("analytical_capacity.json")
    target.write_text(json.dumps(calculate(), indent=2) + "\n")
    print(target.relative_to(ROOT))
