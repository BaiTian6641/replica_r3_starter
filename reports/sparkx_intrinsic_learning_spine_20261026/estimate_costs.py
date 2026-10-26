"""Analytical sizes/MACs for a proposed design; no model code or training."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/sparkx_2_6_4b_intrinsic_learning_spine_v1.json"


def calculate():
    c = json.loads(CONFIG.read_text())
    state = c["state"]
    f, a, p, e, w = (state[k] for k in ("F", "A", "P", "E", "C"))
    sites = len(c["intrinsic_memory"]["reader_sites_candidate"])
    d = c["backbone"]["width"]
    b = c["intrinsic_memory"]["reader_width"]
    mib = 1024**2
    f_bytes = f["depth_groups"] * f["experts_per_group"] * f["heads"] * f["key_width"] * f["value_width"] * 4
    a_bytes = a["groups"] * a["width"]**2 * 4
    per_expert = p["matrices_per_expert"] * p["width"] * p["hidden_width"]
    p_parameters = per_expert * p["experts"]
    p_bytes = p_parameters * 4
    e_bytes = e["spark_records"] * (2 * e["bf16_values_per_record"] + e["metadata_bytes_per_record_cap"])
    c_bytes = (w["workspace_slots"] + w["event_slots"]) * w["width"] * 2
    live = f_bytes + a_bytes + p_bytes + e_bytes + c_bytes
    active_p_bytes = per_expert * p["max_active_experts_per_teaching_event"] * 4
    # Conservative separate F snapshot, A snapshot + gradient, P snapshot + gradient.
    staging = f_bytes + 2 * a_bytes + 2 * active_p_bytes
    reader_parameters = sites * (2 * d * b + 2 * b * b)
    p_projection_parameters = sites * 2 * d * p["width"]
    return {
        "kind": "analytical_lower_bounds_not_measurements",
        "config": str(CONFIG.relative_to(ROOT)),
        "assumptions": [
            "FP32 dynamic matrices/knowledge; BF16 fixed projections and workspace",
            "One global P bank shared by active sites; two active experts per site/event",
            "Initial inner optimizer is SGD with no momentum",
            "Record metadata is a proposed hard cap, not measured Python-object overhead",
            "No double counting of tied/shared parameters",
        ],
        "fixed_parameters": {
            "memory_reader_matrices": reader_parameters,
            "memory_reader_bf16_mib": reader_parameters * 2 / mib,
            "P_input_output_projections": p_projection_parameters,
            "P_projection_bf16_mib": p_projection_parameters * 2 / mib,
        },
        "offline_joint_training_additions": {
            "existing_siglip2_spear_parameters": 92_884_224 + 93_289_476,
            "encoders_fp32_weight_gradient_two_moments_gib": (92_884_224 + 93_289_476) * 16 / 1024**3,
            "activations_and_extensions_included": False,
            "full_multimodal_finetuning_required": True,
            "online_self_learning_substitutes_for_modality_training": False,
        },
        "dynamic_state": {
            "F_mib": f_bytes / mib, "A_mib": a_bytes / mib,
            "P_parameters": p_parameters, "P_fp32_mib": p_bytes / mib,
            "P_full_bank_adam_moments_extra_mib": 2 * p_bytes / mib,
            "E_tensor_mib": e["spark_records"] * e["bf16_values_per_record"] * 2 / mib,
            "E_metadata_cap_mib": e["spark_records"] * e["metadata_bytes_per_record_cap"] / mib,
            "workspace_event_mib": c_bytes / mib,
            "live_lower_bound_mib": live / mib,
            "update_staging_extra_mib": staging / mib,
            "update_peak_lower_bound_mib": (live + staging) / mib,
            "replay_separate_mib": c["online_teaching"]["replay_bytes_cap"] / mib,
            "update_peak_plus_replay_lower_bound_mib": (live + staging + c["online_teaching"]["replay_bytes_cap"]) / mib,
        },
        "P_capacity_growth_candidates": [
            {"experts": count, "FP32_weight_mib": count * per_expert * 4 / mib,
             "active_experts": p["active_experts_per_site"], "admitted": False}
            for count in (16, 64, 128)
        ],
        "matrix_macs": {
            "P_per_processed_vector_two_experts": per_expert * p["active_experts_per_site"],
            "P_nine_sites_per_decoder_position": per_expert * p["active_experts_per_site"] * sites,
            "P_local_forward_128_vectors": per_expert * p["max_active_experts_per_teaching_event"] * 128,
            "backward_cost_included": False,
        },
        "observer_payload": {
            "all_hidden_layers_8192_positions_bf16_gib": c["backbone"]["layers"] * 8192 * d * 2 / 1024**3,
            "nine_sites_eight_512_slots_bf16_kib": sites * c["spine"]["default_observer_slots_per_site"] * c["spine"]["feature_width"] * 2 / 1024,
        },
        "exclusions": [
            "Original backbone, encoders, controller/norm/gate parameters and optimizer state",
            "Activations, KV, routing, indexes, phrase rows and lexical/provenance overflow",
            "Backend scratch, allocator overhead, checkpoint copies and disk serialization",
            "Full prefix/sensing passes, encoder/teacher work, replay and validation compute",
        ],
        "latency_estimated": False, "runtime_admission_granted": False,
    }


if __name__ == "__main__":
    print(json.dumps(calculate(), indent=2))
