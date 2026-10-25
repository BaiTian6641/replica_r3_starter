"""Reproducible partial payload inventory; does not load or execute a model."""
from __future__ import annotations

import json
from pathlib import Path


def calculate() -> dict:
    mib, gib = 2**20, 2**30
    results = {}
    for name, d, ff, layers, vocab, total, entries, tied in (
        ("minicpm5", 2048, 6144, 42, 130560, 2516756480, 1024, False),
        ("sparkx", 2560, 10240, 36, 131072, 4112079360, 4096, True),
    ):
        w, depth, length, rank = 256, 2, 8, 8
        encoder = 2*d*w + depth*12*w*w
        macs = length*d*w + depth*(12*length*w*w + 2*length*length*w) + w*d
        lora = 2*rank*(d+ff)
        bridge = 2*768 + 768*2048 + 2048 + 2048*d + d
        results[name] = {
            "hidden_size": d, "intermediate_size": ff, "layers": layers,
            "vocabulary": vocab, "tied_base_tables": tied,
            "unique_base_parameters": total,
            "base_payload_bf16_gib": total*2/gib,
            "base_payload_fp32_gib": total*4/gib,
            "base_embedding_and_head_unique_parameters": vocab*d*(1 if tied else 2),
            "active_row_examples": [
                {"slots": k, "one_shared_bf16_bank_mib": 2*k*d/mib,
                 "two_bf16_buffers_mib": 4*k*d/mib,
                 "extra_head_macs_per_position": k*d,
                 "extra_head_arithmetic_percent": 100*k/vocab}
                for k in (32, 256, 512, 1024)
            ],
            "phrase_encoder": {
                "width": w, "depth": depth, "example_base_tokens": length,
                "matrix_parameters": encoder,
                "bf16_matrix_weights_mib": encoder*2/mib,
                "fp32_weights_grad_two_adam_moments_mib": encoder*16/mib,
                "matrix_macs_per_phrase": macs,
                "matrix_macs_16_new_phrases": macs*16,
                "matrix_macs_256_new_phrases": macs*256,
            },
            "two_down_projection_lora": {
                "rank": rank, "parameters": lora,
                "bf16_weights_mib": lora*2/mib,
                "fp32_weights_grad_two_adam_moments_mib": lora*16/mib,
            },
            "episodic_payload": {
                "entries": entries, "key_width": 128, "value_width": 256,
                "bf16_keys_values_mib": entries*(128+256)*2/mib,
                "fp32_keys_values_mib": entries*(128+256)*4/mib,
                "metadata_and_surfaces_included": False,
                "lexical_values_share_this_capacity": True,
            },
            "vision_projector_example_parameters": bridge,
            "dvagen_static_clone_counterexample": {
                "extra_parameters_if_two_full_tables_cloned": 2*vocab*d,
                "extra_bf16_weight_mib": 2*vocab*d*2/mib,
                "fp32_weights_grad_two_adam_moments_gib": 2*vocab*d*16/gib,
                "source_inspection_not_execution": True,
                "proposed_implementation_clones_these_tables": False,
            },
        }
    core = 8*256**2 + 2*256*1024 + 24*2*256 + 8*256 + 3*256 + 10*256 + 3
    return {
        "format": "replica.sparkx_consolidation.analytical_costs.v1",
        "kind": "partial_inventory_not_runtime_measurement_or_total_training_budget",
        "units": {"MiB": mib, "GiB": gib, "MAC": "one multiply-accumulate; not one FLOP"},
        "models": results,
        "default_core_parameters": core,
        "default_core_workspace_bf16_kib": 8*256*2/1024,
        "frozen_vision_encoder": {
            "parameters_from_local_v0_report": 92884224,
            "bf16_weight_mib": 92884224*2/mib,
            "fp32_weight_mib": 92884224*4/mib,
            "qualified_v0_compute_dtype": "float32",
        },
        "formulas": {
            "encoder_matrices": "2*d*w + depth*12*w*w",
            "phrase_macs": "L*d*w + depth*(12*L*w*w + 2*L*L*w) + w*d",
            "two_lora_down_projections": "2*r*(d+ff)",
            "fp32_adam_payload": "16 bytes/parameter: weight, gradient, two moments; no extra master copy",
            "bridge": "LayerNorm(768) + Linear(768,2048,bias) + Linear(2048,d,bias)",
        },
        "exclusions": [
            "Encoder norms/biases and new selector, writer, reader and learning-policy parameters",
            "Full logits, KV, attention/saved activations, gradients through the frozen backbone",
            "Optimizer workspaces, master copies if used, allocator/transient buffers",
            "Surface bytes, provenance, revision history, bounded replay and storage I/O",
            "Optional static-pattern tables, fast-memory ports, temporal module/resampler",
            "Repeated per-session active vocab banks and visual encodings",
            "Any latency, quality or total-training-memory inference from these payloads",
        ],
    }


if __name__ == "__main__":
    target = Path(__file__).with_name("analytical_costs.json")
    target.write_text(json.dumps(calculate(), indent=2) + "\n")
    print(target)
