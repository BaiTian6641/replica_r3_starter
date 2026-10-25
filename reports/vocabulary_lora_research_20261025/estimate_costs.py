"""Analytical budgets for the vocabulary/LoRA proposal; no model execution."""
from __future__ import annotations

import json
import math
from pathlib import Path


def calculate() -> dict:
    d, vocab, layers = 2560, 131072, 36
    mib = 2 ** 20
    projections = {
        "q_k_v_proj": (2560, 6144),
        "out_proj": (4096, 2560),
        "gate_proj": (2560, 10240),
        "up_proj": (2560, 10240),
        "down_proj": (10240, 2560),
    }
    vocab_rows = []
    for k in (256, 1024, 2048):
        p = k * d
        vocab_rows.append({
            "new_rows": k, "parameters": p,
            "bf16_weights_mib": 2 * p / mib,
            "fp32_weights_grad_two_adam_moments_mib": 16 * p / mib,
            "extra_output_head_macs_per_position": p,
            "extra_output_head_compute_percent": 100 * k / vocab,
            "mean_row_initial_old_distribution_kl_bound_nats": math.log1p(k / vocab),
        })
    lora = []
    for rank in (8, 16):
        for label, target_list in (
            ("two_square_residual_ports", [(d, d)] * 2),
            ("two_backbone_down_projections", [projections["down_proj"]] * 2),
            ("eight_blocks_five_projections", list(projections.values()) * 8),
            ("all_36_blocks_five_projections", list(projections.values()) * layers),
        ):
            a = sum(rank * din for din, _ in target_list)
            b = sum(rank * dout for _, dout in target_list)
            lora.append({
                "scope": label, "rank": rank,
                "a_parameters": a, "b_parameters": b, "total_parameters": a + b,
                "bf16_all_adapter_weights_mib": 2 * (a + b) / mib,
                "fp32_weights_grad_two_adam_moments_mib": 16 * (a + b) / mib,
                "fa_b_only_fp32_weights_grad_two_adam_moments_mib": 16 * b / mib,
                "fa_frozen_a_bf16_mib": 2 * a / mib,
                "unmerged_adapter_forward_macs_per_position": a + b,
            })
    activation = []
    for n in (1, 32, 256, 2048):
        r = 16
        # Conservative comparison to the inspected CARE implementation:
        # vanilla saves BF16 X and Z; CARE retains FP32 Z and M.
        vanilla = 2 * n * (d + r)
        fa = 2 * n * r
        care = 4 * r * (n + d)
        activation.append({
            "flattened_batch_positions": n, "input_width": d, "rank": r,
            "vanilla_bf16_x_and_z_bytes": vanilla,
            "fa_bf16_z_bytes": fa,
            "care_fp32_z_and_m_bytes": care,
            "care_over_vanilla_saved_branch_payload_ratio": care / vanilla,
        })
    return {
        "kind": "analytical_estimate_not_device_measurement",
        "units": {"MiB": "2^20 bytes", "MAC": "one multiply-accumulate"},
        "spark": {"hidden_size": d, "vocab_size": vocab, "num_hidden_layers": layers,
                  "tie_word_embeddings": True,
                  "base_tied_embedding_parameters": d * vocab,
                  "base_tied_embedding_bf16_mib": 2 * d * vocab / mib,
                  "linear_input_output_shapes": projections},
        "vocabulary": vocab_rows,
        "lora": lora,
        "saved_activation_examples": activation,
        "care_fp32_payload_break_even_positions_strictly_above": 2 * 16 * d / (d - 16),
        "fp32_covariance_mib": {str(w): 4 * w * w / mib for w in (2560, 4096, 10240)},
        "selected_16_alias_embeddings_bf16_kib": 16 * d * 2 / 1024,
        "exclusions": [
            "Backbone, vision and existing memory weights except the separately listed embedding table",
            "Attention/KV, other saved activations, full logits, temporary tensors, allocator overhead",
            "CARE solve workspaces, other retained tensors and dtype conversions",
            "FA preconditioner and optimizer implementation workspaces",
            "Tokenizer strings, indices, provenance, replay, data loading and compilation",
            "Wall time, device peak memory, power and quality are unmeasured",
        ],
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("analytical_costs.json")
    out.write_text(json.dumps(calculate(), indent=2) + "\n")
    print(out)
