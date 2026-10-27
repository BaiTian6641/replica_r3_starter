"""Reproduce v5 architecture arithmetic only; no ML imports or native execution."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MIB, GIB = 1024**2, 1024**3


def calculate():
    c = json.loads((ROOT / "configs/sparkx_2_6_4b_final_product_v5.json").read_text())
    b, v, a = c["backbone"], c["vision"], c["audio"]
    p, core = c["plastic"], c["core"]
    vocab, ep, integ = c["vocabulary"], c["episodic"], c["intrinsic_integration"]
    sites, width = len(integ["spark_sites"]), b["width"]

    def affine_pair(widths):
        x, y, z = widths
        return x * y + y + y * z + z

    counts = {
        "original_backbone": b["unique_parameters"],
        "vision_encoder_declared": v["declared_parameters"],
        "audio_encoder_declared": a["declared_parameters"],
        "vision_projector_with_bias": affine_pair(v["projector_widths"]),
        "audio_projector_with_bias": affine_pair(a["projector_widths"]),
        "nine_compact_readers": sites * sum(x * y for x, y in integ["reader_matrices_per_site"].values()),
        "nine_plastic_projections": sites * 2 * width * p["width"],
        "two_projection_phrase_encoder": vocab["encoder_layers"] * vocab["matrix_terms_per_encoder_layer"] * vocab["encoder_width"] ** 2 + vocab["input_and_output_projection_count"] * width * vocab["encoder_width"],
    }
    plastic_count = p["experts_stored"] * p["matrices_per_expert"] * p["width"] * p["hidden_width"]
    stable = sum(counts.values())
    combined = stable + plastic_count
    assert stable == c["resource_profile"]["partial_stable_parameters"]
    assert combined == c["resource_profile"]["partial_stable_plus_plastic_parameters"]

    old = json.loads((ROOT / "reports/sparkx_residency_20261026/residency_v2.json").read_text())["parameter_inventory"]
    old_terms = {r["id"]: r["parameters"] for r in old["components"]}
    removed = {k: old_terms[k] for k in c["inventory_reconciliation"]["remove_alternative_terms"]}
    added = {k: counts[k] for k in ["nine_compact_readers", "nine_plastic_projections", "two_projection_phrase_encoder"]}
    added["sixteen_plastic_experts"] = plastic_count
    assert old["total_parameters"] - sum(removed.values()) + sum(added.values()) == combined

    f = c["fast_memory"]["spark"]
    addr = c["address_adaptation"]
    e_bytes = ep["spark_records"] * (ep["feature_elements_per_record"] * 2 + ep["metadata_bytes_cap_per_record"])
    f_bytes = f["groups"] * f["experts_per_group"] * f["heads"] * f["head_width"]**2 * 4
    core_bytes = (core["workspace_slots"] + core["event_slots"]) * core["width"] * 2
    p_bytes = plastic_count * 4
    a_bytes = addr["spark_matrices"] * addr["width"]**2 * 4
    selected_bytes = p["updated_experts_per_transaction_max"] * p["matrices_per_expert"] * p["width"] * p["hidden_width"] * 4
    live = {"E": e_bytes, "F": f_bytes, "C": core_bytes, "P": p_bytes, "P_moments": 2*p_bytes}
    optional_a = {"A": a_bytes, "A_moments": 2*a_bytes}
    staging = {"F_candidate": f_bytes, "C_candidate": core_bytes, "one_E_record": e_bytes // ep["spark_records"], "selected_P_candidate": selected_bytes, "selected_P_candidate_moments": 2*selected_bytes, "selected_P_gradients": selected_bytes}
    optional_a_staging = {"A_candidate": a_bytes, "A_candidate_moments": 2*a_bytes, "A_gradients": a_bytes}
    live_with_a = sum(live.values()) + sum(optional_a.values())
    peak_with_a = live_with_a + sum(staging.values()) + sum(optional_a_staging.values())
    prior = json.loads((ROOT / "reports/sparkx_nested_learning_20261026/analytical_costs.json").read_text())["profiles"]["spark"]
    assert live_with_a / MIB == prior["with_adaptive_address_live_MiB"]
    assert peak_with_a / MIB == prior["illustrated_peak_with_adaptive_address_MiB"]

    kv_unit = 2 * b["kv_heads"] * b["head_width"] * 2
    ctx = b["initial_active_context_positions"]
    full_layers = len(b["full_attention_layers"])
    bounded_kv = kv_unit * (full_layers * ctx + b["sliding_layers_count"] * min(ctx, b["sliding_window"]))
    full_kv = kv_unit * b["layers"] * ctx
    rp = c["resource_profile"]
    n = rp["rough_planning_parameters"]
    av = rp["AV_example"]
    av_count = (av["frames"] + av["crops"]) * av["tokens_per_view"] + av["audio_tokens"] + av["event_positions"] + av["clock_positions"]
    assert av_count == av["total_positions"]
    phrase_bytes = vocab["active_phrase_slots"] * width * 2
    observer = c["spine"]["observer_slots_per_site"] * c["spine"]["feature_width"] * sites * 2

    return {
        "kind": "analytical_payload_account_not_measured_fit",
        "config": "configs/sparkx_2_6_4b_final_product_v5.json",
        "native_execution": False,
        "training": False,
        "parameters": {"stable_components": counts, "stable_partial": stable, "plastic": plastic_count, "combined_partial": combined, "unsealed_modules": c["inventory_reconciliation"]["unsealed_modules"], "rough_planning_placeholder_not_cap": n},
        "reconciliation": {"old_profile_total": old["total_parameters"], "remove_alternative_terms": removed, "add_reference_terms": added, "new_partial": combined, "historical_receipts_modified": False},
        "dynamic": {"live_bytes": live, "optional_A_bytes": optional_a, "staging_bytes": staging, "optional_A_staging_bytes": optional_a_staging, "live_with_A_MiB": live_with_a/MIB, "peak_with_A_MiB": peak_with_a/MIB, "peak_with_A_plus_separate_replay_MiB": peak_with_a/MIB + rp["separate_replay_envelope_MiB"], "matches_prior_nested_payload": True, "128_MiB_live_profile_fits": live_with_a <= 128*MIB},
        "KV": {"bytes_per_token_per_layer_BF16": kv_unit, "bounded_sliding_cache_bytes": bounded_kv, "bounded_sliding_cache_MiB": bounded_kv/MIB, "all_layers_full_cache_bytes": full_kv, "all_layers_full_cache_GiB": full_kv/GIB, "backend_eviction_verified": False},
        "training_state_GiB_at_planning_placeholder": {"16_bytes_per_parameter": n*16/GIB, "18_bytes_per_parameter": n*18/GIB, "12_bytes_per_parameter_unqualified": n*12/GIB},
        "derived_payloads": {"phrase_rows_MiB": phrase_bytes/MIB, "double_phrase_buffer_MiB": 2*phrase_bytes/MIB, "phrase_head_extra_percent": 100*vocab["active_phrase_slots"]/b["base_vocab"], "bounded_observer_KiB": observer/1024, "full_hidden_dump_GiB": b["layers"]*ctx*width*2/GIB, "AV_example_positions": av_count, "selected_record_KV_per_site_KiB": integ["selected_record_packets_max"]*integ["reader_width"]*2*2/1024, "selected_record_KV_all_sites_MiB": sites*integ["selected_record_packets_max"]*integ["reader_width"]*2*2/MIB, "two_active_P_matrix_MACs_per_position_per_site": p["active_experts_per_decision_max"]*p["matrices_per_expert"]*p["width"]*p["hidden_width"]},
        "excluded": ["unsealed module weights", "activations", "encoder internal work buffers", "allocator and kernel temporaries", "additional candidate or rollback snapshots", "optimizer scalar metadata", "source/literal payloads", "expanded journal and indexes", "multi-stream multiplication", "reference teacher models", "external modules"]
    }


if __name__ == "__main__":
    result = calculate()
    (OUT / "analytical_costs.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stable_partial": result["parameters"]["stable_partial"], "combined_partial": result["parameters"]["combined_partial"], "dynamic_peak_with_A_MiB": result["dynamic"]["peak_with_A_MiB"], "native_execution": False}))
